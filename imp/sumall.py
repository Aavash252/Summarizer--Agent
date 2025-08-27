from typing import List, Dict
from pydantic import BaseModel, Field
from functools import partial

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_ollama import ChatOllama

from tools import get_text_from_url  # your existing helper


# ---------- Models ----------
class ArticleFourPointSummary(BaseModel):
    summary_points: List[str] = Field(
        description="Exactly 4 concise, objective bullet points"
    )

# Optional: final synthesis across sources
class FourPointSynthesis(BaseModel):
    summary_points: List[str] = Field(
        description="Exactly 4 concise, objective bullets consolidating multiple articles"
    )


# ---------- LLMs ----------
# Use JSON mode to avoid invalid JSON; keep temperature 0 for determinism.
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)
parser4 = JsonOutputParser(pydantic_object=ArticleFourPointSummary)

synth_parser4 = JsonOutputParser(pydantic_object=FourPointSynthesis)
synth_llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)


# ---------- Prompts ----------
summ_system = """You are a professional news summarizer.
Write EXACTLY 4 concise, objective bullet points capturing the key facts and developments.
No opinions. No duplication. Short lines. Output JSON only."""
summ_user_tmpl = """ARTICLE TEXT:
```{article_text}```"""

summ_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summ_system),
        ("human", summ_user_tmpl),
    ]
)

summarizer_chain = summ_prompt | llm_json | parser4


# Optional synthesis across the 4 articles
synth_system = """You are consolidating bullet points from multiple news articles.
Return EXACTLY 4 objective, deduplicated bullets that cover the most important, non-overlapping facts.
No opinions. Short lines. Output JSON only."""
synth_user_tmpl = """Here are bullet lists from multiple sources:
{all_points}

Return ONLY JSON with:
- "summary_points": list of 4 bullets."""

synth_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", synth_system),
        ("human", synth_user_tmpl),
    ]
)

synthesis_chain = synth_prompt | synth_llm_json | synth_parser4


# ---------- Helpers ----------
def _fetch_and_pack(url: str, _=None) -> Dict[str, str]:
    """
    RunnableLambda-friendly function:
    returns dict for the summarizer_chain's input.
    """
    text = get_text_from_url(url)
    return {"article_text": text}

def make_url_agent(url: str):
    """
    Build a runnable for one URL: fetch → summarize(4 bullets)
    """
    return RunnableLambda(partial(_fetch_and_pack, url)) | summarizer_chain


# ---------- Public API ----------
def analyze_multiple_articles(urls: List[str], do_synthesis: bool = True) -> Dict:
    """
    Run N URL agents in parallel (expecting 4), each outputting 4 bullets.
    Optionally synthesize across them into 4 final bullets.
    """
    if not urls:
        raise ValueError("Provide at least one URL.")
    # Build a dict of runnables keyed by source_i for RunnableParallel
    agents = {f"source_{i+1}": make_url_agent(u) for i, u in enumerate(urls)}

    parallel = RunnableParallel(**agents)
    per_source = parallel.invoke({})  # {'source_1': {'summary_points': [...]}, ...}
    print(per_source)

    # Flatten for convenience and prepare for optional synthesis
    all_points_by_source = {
        k: v["summary_points"] for k, v in per_source.items()
    }

    result = {
        "per_source_summaries": all_points_by_source
    }

    if do_synthesis:
        # Prepare synth input: a compact, numbered block of bullets
        blocks = []
        for i, (k, bullets) in enumerate(all_points_by_source.items(), start=1):
            btxt = "\n".join([f"- {b}" for b in bullets])
            blocks.append(f"Source {i} ({k}):\n{btxt}")
        all_points_text = "\n\n".join(blocks)

        final = synthesis_chain.invoke({"all_points": all_points_text})
        result["consolidated_4_points"] = final["summary_points"]

    return result


# ---------- Example usage ----------
if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
        "https://www.bbc.com/sport/football/live/cn5epw6vqyzt",
        "https://www.bbc.com/sport/football/articles/cgjye1144l2o",
        "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno",
    ]
    output = analyze_multiple_articles(urls, do_synthesis=True)
    # output["per_source_summaries"] => dict of 4-bullet lists per URL
    # output["consolidated_4_points"] => 4-bullet synthesis across all sources
    print(output)

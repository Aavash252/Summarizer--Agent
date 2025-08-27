# --- imports ---
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_url

# --- shared LLM (JSON mode) ---
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# ---------- Agent: per-article single key point ----------
class OnePointOut(BaseModel):
    key_point: str = Field(description="ONE concise, headline-style bullet point for this article.")

system_prompt = """You are a professional news summarizer.
Write EXACTLY ONE objective, headline-style key point in few words.

Rules:
- Be factual and neutral.
- Prioritize concrete outcomes, numbers, names, places, and dates.
- Use only information in the provided text.
- Use absolute dates as written; do not normalize.
- No quotes, no emojis, no hashtags, no links, no markdown bullets.

DON'Ts:
- Don't change any NUMBERS 
- Don't write more than one sentence.
- Don't include multiple unrelated facts.
- Don't add opinion or analysis.
- Don't add any extra keys or text outside JSON.

Return ONLY valid JSON:
{{"key_point": "<your single concise point>"}}"""


onept_user = "ARTICLE TEXT:```{article_text}```"

onept_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", onept_user),
])

onept_parser = JsonOutputParser(pydantic_object=OnePointOut)
one_point_chain = onept_prompt | llm_json | onept_parser

# ---------- Helper to summarize one URL ----------
def summarize_url_to_one_point(url: str) -> dict:
    data = get_text_from_url(url)
    text = (data or {}).get("text", "")
    if not text:
        # graceful fallback to avoid whole run failing
        return {"key_point": f"(No text extracted from {url})"}
    try:
        out = one_point_chain.invoke({"article_text": text})
        # out is a Pydantic model OR dict depending on lc version; normalize:
        if hasattr(out, "key_point"):
            return {"key_point": out.key_point}
        return {"key_point": out["key_point"]}
    except Exception as e:
        return {"key_point": f"(Summarization failed for {url}: {e})"}

# ---------- Build a parallel runnable for N urls ----------
def build_parallel_for_urls(urls: list[str]) -> RunnableParallel:
    branches = {}
    for i, u in enumerate(urls, start=1):
        branches[f"news{i}"] = RunnableLambda(lambda _x, _u=u: summarize_url_to_one_point(_u))
    print(branches)
    return RunnableParallel(**branches)

class MergeOut(BaseModel):
    summary_points: list[str] = Field(description="Exactly 4 bullets, one per input news, in original order.")

merge_system = """You are an editor.
Merge the four single-article key points into EXACTLY FOUR bullets, preserving input order (news1..news4).
Do not re-write or embellish beyond minimal cleanup.
Return ONLY:
{{"summary_points": ["...", "...", "...", "..."]}}"""

merge_user = """news1: {n1}
news2: {n2}
news3: {n3}
news4: {n4}"""

merge_prompt = ChatPromptTemplate.from_messages([
    ("system", merge_system),
    ("user", merge_user),
])

merge_parser = JsonOutputParser(pydantic_object=MergeOut)
merge_chain = merge_prompt | llm_json | merge_parser

# ---------- Orchestration ----------
def summarize_four_urls(urls: list[str]) -> dict:
    if len(urls) != 4:
        raise ValueError("Provide exactly 4 URLs.")
    print("Article content fetched successfully. Analyzing...")
    parallel = build_parallel_for_urls(urls)
    tri = parallel.invoke({})   #Run all summarizations at once , wait until all are finished and return the output.

    # Preserve input order explicitly
    ordered_points = [
        tri["news1"]["key_point"],
        tri["news2"]["key_point"],
        tri["news3"]["key_point"],
        tri["news4"]["key_point"],
    ]

    # Optional: pass through editor for validation/cleanup
    merged = merge_chain.invoke({
        "n1": ordered_points[0],
        "n2": ordered_points[1],
        "n3": ordered_points[2],
        "n4": ordered_points[3],
    })

    # Normalize return (pydantic/dict)
    final_points = merged.summary_points if hasattr(merged, "summary_points") else merged["summary_points"]
    return {"summary_points": final_points}

# ---------- Example main ----------
if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
        "https://www.bbc.com/sport/football/articles/c4gljqwe5g8o",
        "https://www.bbc.com/sport/football/articles/crmvygekeyzo",
        "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno", 
    ]
    result = summarize_four_urls(urls)
    print("\nFOUR-NEWS SUMMARY:")
    for p in result["summary_points"]:
        print(f"- {p}")

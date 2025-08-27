# --- imports ---
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from tools import get_text_from_url
from datetime import datetime

from tools import get_text_from_url  # <- your existing fetcher

# ---------------- Shared LLM (JSON mode) ----------------
# You can use different models per agent if desired.
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# =============== Agent 1: Summarizer ====================
class SummaryOut(BaseModel):
    summary_points: List[str] = Field(description="Exactly 3 concise bullets.")

sum_sys = (
    "You are a professional news summarizer.\n"
    "Write EXACTLY 3 objective, headline-style bullet points.\n"
    "No opinions, no analysis, no duplicates, no preamble."
)
sum_user = "ARTICLE TEXT:```{article_text}```"
sum_prompt = ChatPromptTemplate.from_messages([("system", sum_sys), ("user", sum_user)])
sum_parser = JsonOutputParser(pydantic_object=SummaryOut)
summarizer_chain: Runnable[Dict[str, Any], SummaryOut] = (
    sum_prompt | llm_json | sum_parser
)

# =============== Agent 2: Sentiment =====================
class SentimentOut(BaseModel):
    sentiment: str = Field(description="One of: Positive, Negative, Neutral")

sent_sys = (
    "Classify overall sentiment of the article content.\n"
    "Return only one of: Positive, Negative, Neutral.\n"
    "If mixed/unclear -> Neutral."
)
sent_user = "ARTICLE TEXT:```{article_text}```"
sent_prompt = ChatPromptTemplate.from_messages([("system", sent_sys), ("user", sent_user)])
sent_parser = JsonOutputParser(pydantic_object=SentimentOut)
sentiment_chain: Runnable[Dict[str, Any], SentimentOut] = (
    sent_prompt | llm_json | sent_parser
)

# =============== Agent 3: Category ======================
class CategoryOut(BaseModel):
    category: str = Field(
        description="One of: Political, Business, Sports, Entertainment, Scientific, General"
    )

cat_sys = (
    "Assign the best-fitting category for the article.\n"
    "Choose one of: Political, Business, Sports, Entertainment, Scientific, General.\n"
    "If unclear -> General."
)
cat_user = "ARTICLE TEXT:```{article_text}```"
cat_prompt = ChatPromptTemplate.from_messages([("system", cat_sys), ("user", cat_user)])
cat_parser = JsonOutputParser(pydantic_object=CategoryOut)
category_chain: Runnable[Dict[str, Any], CategoryOut] = (
    cat_prompt | llm_json | cat_parser
)

# ============ Editor/Merger Agent (recommended) =========
class EditorOut(BaseModel):
    summary_points: List[str]
    sentiment: str
    category: str

editor_sys = (
    "You are an editor who merges specialist agents' outputs into one clean JSON.\n"
    "- Keep summary_points exactly as provided (unless they contain opinion or duplication—then minimally fix).\n"
    "- Ensure sentiment is one of: Positive, Negative, Neutral.\n"
    "- Ensure category is one of: Political, Business, Sports, Entertainment, Scientific, General.\n"
    "Return only JSON with keys: summary_points, sentiment, category."
)
editor_user = """
TITLE: {title}
PUBLISHED: {published}
SPECIALIST_OUTPUTS:
- SUMMARY: {summary_json}
- SENTIMENT: {sentiment_json}
- CATEGORY: {category_json}
"""
editor_prompt = ChatPromptTemplate.from_messages([("system", editor_sys), ("user", editor_user)])
editor_parser = JsonOutputParser(pydantic_object=EditorOut)
editor_chain: Runnable[Dict[str, Any], EditorOut] = editor_prompt | llm_json | editor_parser

# =================== Utilities ==========================
def format_publish_date(iso_date: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error formatting date: {e}"

# Optional: small helper to clamp/validate specialist outputs
def _normalize_specialist_outputs(tri_output: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce/validate the 3 specialist outputs; raise if invalid."""
    if not isinstance(tri_output.get("summary"), SummaryOut):
        raise ValueError("Summary branch returned unexpected type.")
    if not isinstance(tri_output.get("sentiment"), SentimentOut):
        raise ValueError("Sentiment branch returned unexpected type.")
    if not isinstance(tri_output.get("category"), CategoryOut):
        raise ValueError("Category branch returned unexpected type.")
    return tri_output

# ================= Orchestration ========================
# Notes:
# - We add .with_retry() to each branch to smooth over transient LLM hiccups.
# - We add timeouts around the parallel call; any failure falls back to raw outputs if possible.
def make_parallel() -> RunnableParallel:
    return RunnableParallel(
        summary=summarizer_chain.with_retry(stop_after_attempt=2),
        sentiment=sentiment_chain.with_retry(stop_after_attempt=2),
        category=category_chain.with_retry(stop_after_attempt=2),
    )

def analyze_article(article_text: str, title: str = "", published: str = "") -> Dict[str, Any]:
    if not article_text or not article_text.strip():
        raise ValueError("Empty article_text.")

    parallel = make_parallel()

    # You can set a soft timeout for the whole parallel run:
    # (timeouts are advisory; actual enforcement depends on the underlying executors)
    tri_output = parallel.invoke({"article_text": article_text}, config={"timeout": 60})

    # Validate/coerce specialist outputs
    tri_output = _normalize_specialist_outputs(tri_output)

    # Send to Editor for final polish/validation
    try:
        merged = editor_chain.invoke(
            {
                "title": title or "Untitled",
                "published": published or "Unknown",
                "summary_json": tri_output["summary"].model_dump_json(),
                "sentiment_json": tri_output["sentiment"].model_dump_json(),
                "category_json": tri_output["category"].model_dump_json(),
            },
            config={"timeout": 45},
        )
        return merged.model_dump()

    except (ValidationError, Exception):
        # Fallback: return raw specialist outputs if Editor fails
        return {
            "summary_points": tri_output["summary"].summary_points,
            "sentiment": tri_output["sentiment"].sentiment,
            "category": tri_output["category"].category,
            "_note": "Editor merge failed; returning raw specialist outputs.",
        }

def analyze_url(url: str) -> Dict[str, Any]:
    print(f"-> Fetching article content from: {url}")
    article_data = get_text_from_url(url)
    title = article_data.get("title", "") or "Untitled"
    text = article_data.get("text", "") or ""
    date = article_data.get("date", "") or ""
    if not text:
        raise RuntimeError("Could not retrieve article text.")

    published = format_publish_date(date) if date else "Unknown"
    print("-> Article content fetched successfully. Analyzing...")
    return analyze_article(text, title, published)

def pretty_print_result(title: str, published: str, result: Dict[str, Any]) -> None:
    print("\n-------------------")
    print("Analysis Complete:")
    print(f"TITLE: {title}")
    print(f"Published Date: {published}")
    print(f"CATEGORY: {result['category']}")
    print("\nSUMMARY:")
    for p in result["summary_points"]:
        print(f"- {p}")
    print(f"\nSENTIMENT: {result['sentiment']}")
    print("-------------------\n")

def analyze_many(urls: List[str]) -> List[Dict[str, Any]]:
    results = []
    for u in urls:
        try:
            data = get_text_from_url(u)
            title = data.get("title", "") or "Untitled"
            text = data.get("text", "") or ""
            date = data.get("date", "") or ""
            published = format_publish_date(date) if date else "Unknown"
            merged = analyze_article(text, title, published)
            results.append({"url": u, "title": title, "published": published, **merged})
        except Exception as e:
            results.append({"url": u, "error": str(e)})
    return results

if __name__ == "__main__":
    target_url = "https://www.bbc.com/sport/football/articles/cx2q009xzgeo"
    try:
        article_data = get_text_from_url(target_url)
        title = article_data.get("title", "") or "Untitled"
        text = article_data.get("text", "") or ""
        date = article_data.get("date", "") or ""
        if not text:
            raise RuntimeError("-> Could not retrieve article text. Exiting.")

        frt_date = format_publish_date(date) if date else "Unknown"
        print("-> Article content fetched successfully. Analyzing...")
        result = analyze_article(text, title, frt_date)
        # Print a friendly report
        pretty_print_result(title, frt_date, result)

        # Or just print JSON:
        # import json; print(json.dumps({"title": title, "published": frt_date, **result}, indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")

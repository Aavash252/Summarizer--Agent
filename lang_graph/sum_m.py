# --- imports ---
import os
import time
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_file,write_summary_to_file

# --- shared LLM (JSON mode) ---
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# ---------- Agent: Per-Article Single Key Point ----------
class OnePointOut(BaseModel):
    key_point: str = Field(description="ONE concise, headline-style bullet point for this article.")

onept_system_prompt = """You are a professional news summarizer.
Write EXACTLY ONE objective, headline-style key point in few words.
Return ONLY valid JSON: {{"key_point": "<your single concise point>"}}"""
onept_user = "ARTICLE TEXT:```{article_text}```"
onept_prompt = ChatPromptTemplate.from_messages([("system", onept_system_prompt), ("user", onept_user)])

onept_parser = JsonOutputParser(pydantic_object=OnePointOut)
one_point_chain = onept_prompt | llm_json | onept_parser

# ---------- Agent: Per-Article Category ----------
class CategoryOut(BaseModel):
    category: str = Field(description="The primary category of the news article (e.g., 'Sports', 'Politics', 'Technology').")

category_system_prompt = """You are a news classifier.
From the article text, identify the single most relevant category (e.g., Sports, Politics, Business, Technology).
Return ONLY valid JSON: {{"category": "<your category>"}}"""
category_user = "ARTICLE TEXT:```{article_text}```"

category_prompt = ChatPromptTemplate.from_messages([("system", category_system_prompt), ("user", category_user)])
category_parser = JsonOutputParser(pydantic_object=CategoryOut)
category_chain = category_prompt | llm_json | category_parser

# ---------- Agent: Per-Article Sentiment ----------
class SentimentOut(BaseModel):
    sentiment: str = Field(description="The overall sentiment of the article (e.g., 'Positive', 'Negative', 'Neutral').")

sentiment_system_prompt = """You are a sentiment analyst.
Determine the overall sentiment of the news article (Positive, Negative, or Neutral).
Return ONLY valid JSON: {{"sentiment": "<your sentiment>"}}"""
sentiment_user = "ARTICLE TEXT:```{article_text}```"
sentiment_prompt = ChatPromptTemplate.from_messages([("system", sentiment_system_prompt), ("user", sentiment_user)])
sentiment_parser = JsonOutputParser(pydantic_object=SentimentOut)
sentiment_chain = sentiment_prompt | llm_json | sentiment_parser


# ---------- Helper to analyze one file with all agents ----------
def analyze_file(filepath: str) -> dict:
    text = get_text_from_file(filepath)
    if not text:
        return {"key_point": f"(No text found in {filepath})", "category": "Unknown", "sentiment": "Unknown"}
    try:
        analysis_branches = RunnableParallel(key_point=one_point_chain, category=category_chain, sentiment=sentiment_chain)
        result = analysis_branches.invoke({"article_text": text})

        # Robustly extract data whether the parser returns a Pydantic model or a dict
        key_point_obj = result["key_point"]
        category_obj = result["category"]
        sentiment_obj = result["sentiment"]

        return {
            "key_point": key_point_obj.key_point if hasattr(key_point_obj, 'key_point') else key_point_obj['key_point'],
            "category": category_obj.category if hasattr(category_obj, 'category') else category_obj['category'],
            "sentiment": sentiment_obj.sentiment if hasattr(sentiment_obj, 'sentiment') else sentiment_obj['sentiment']
        }
    except Exception as e:
        return {"key_point": f"(Analysis failed for {filepath}: {e})", "category": "Error", "sentiment": "Error"}

# ---------- Build a parallel runnable for N files ----------
def build_parallel_for_files(filepaths: list[str]) -> RunnableParallel:
    branches = {}
    for i, fpath in enumerate(filepaths, start=1):
        branches[f"news{i}"] = RunnableLambda(lambda _x, _fp=fpath: analyze_file(_fp))
    print(f"Parallel branches constructed for: {filepaths}")
    return RunnableParallel(**branches)

# ---------- Agent: Merge all results ----------
class MergedArticle(BaseModel):
    key_point: str
    category: str
    sentiment: str

class MergeOut(BaseModel):
    summary_points: list[MergedArticle] = Field(description="Exactly 4 article summaries, one per input, in original order.")

merge_system = """You are an editor.
Merge the four single-article analysis results into a structured list of EXACTLY FOUR objects.
Preserve the input order (news1..news4). Do not re-write or embellish the content.

Return ONLY a single valid JSON object with one key, "summary_points", which contains a list of the four objects.

EXAMPLE OUTPUT FORMAT:
{{
  "summary_points": [
    {{"key_point": "...", "category": "...", "sentiment": "..."}},
    {{"key_point": "...", "category": "...", "sentiment": "..."}},
    {{"key_point": "...", "category": "...", "sentiment": "..."}},
    {{"key_point": "...", "category": "...", "sentiment": "..."}}
  ]
}}
"""

merge_user = """news1: {n1}
news2: {n2}
news3: {n3}
news4: {n4}"""

merge_prompt = ChatPromptTemplate.from_messages([("system", merge_system), ("user", merge_user)])
merge_parser = JsonOutputParser(pydantic_object=MergeOut)
merge_chain = merge_prompt | llm_json | merge_parser



# ---------- Orchestration ----------
def summarize_four_files(filepaths: list[str]) -> dict:
    if len(filepaths) != 4:
        raise ValueError("Provide exactly 4 file paths.")
    print("Analyzing local article files...")
    parallel_runner = build_parallel_for_files(filepaths)
    analysis_results = parallel_runner.invoke({})

    merged = merge_chain.invoke({
        "n1": analysis_results["news1"],
        "n2": analysis_results["news2"],
        "n3": analysis_results["news3"],
        "n4": analysis_results["news4"],
    })


    if isinstance(merged, MergeOut):
        final_points = merged.summary_points
    elif isinstance(merged, dict) and "summary_points" in merged:
        final_points = merged["summary_points"]
    else:
        raise TypeError(
            "The merge chain returned an unexpected data structure. "
            f"Expected a dictionary with a 'summary_points' key, but got: {merged}"
        )

    return {"summary_points": final_points}

# ---------- Example main ----------
if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
        "https://www.bbc.com/sport/football/articles/c4gljqwe5g8o",
        "https://www.bbc.com/sport/football/articles/crmvygekeyzo",
        "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno", 
    ]
    filepaths = [
    "news/news1.txt",
    "news/news2.txt",
    "news/news3.txt",
    "news/news4.txt",
    ]

    if not all(os.path.exists(fp) for fp in filepaths):
        print("Error: One or more specified text files not found.")
        print("Please create article1.txt, article2.txt, article3.txt, and article4.txt.")
    else:
        start_time = time.time()
        result = summarize_four_files(filepaths)
        end_time = time.time()
        print("\n--- FOUR-NEWS SUMMARY FROM LOCAL FILES ---")
        
        for i, p in enumerate(result["summary_points"], 1):
            print(f"\n--- Article {i}---")
            if isinstance(p, dict):
                print(f"- Point:     {p.get('key_point')}")
                print(f"- Category:  {p.get('category')}")
                print(f"- Sentiment: {p.get('sentiment')}")
            else: 
                print(f"- Point:     {p.key_point}")
                print(f"- Category:  {p.category}")
                print(f"- Sentiment: {p.sentiment}")
    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")
    write_summary_to_file(filepaths, result["summary_points"])

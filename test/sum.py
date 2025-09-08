# --- imports ---
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import json

# --- shared LLM (JSON mode) ---
# Note: Ensure you have Ollama installed and the specified model pulled, e.g., `ollama pull qwen2:7b`
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)


# ---------- Agent 1: Search Query Generator ----------

class SearchQueries(BaseModel):
    queries: list[str] = Field(description="A list of 5 diverse search engine queries.")

query_gen_system = """You are an expert search strategist. Based on the user's topic, generate a diverse list of 5 search engine queries to find recent news articles from BBC News.

Rules:
- Create varied queries. Include different keywords, questions, and angles.
- Do NOT include "site:bbc.com" in the queries themselves. The search tool will handle that.
- Focus on terms that would appear in news headlines or summaries.
- For a topic like "Everest Climbers", queries could be "Mount Everest climbing season 2024 news", "new Everest climbing records", "dangers on Everest 2024", "sherpa guides Everest", "Everest death toll update".

Return ONLY a valid JSON object with a single key "queries".
Example:
{{"queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]}}"""

query_gen_user = "TOPIC: ```{topic}```"

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system),
    ("user", query_gen_user),
])

query_gen_parser = JsonOutputParser(pydantic_object=SearchQueries)
generate_queries_chain = query_gen_prompt | llm_json | query_gen_parser

def generate_search_queries(topic: str) -> list[str]:
    """Uses an LLM to generate a list of search queries for a given topic."""
    print("Generating diverse search queries...")
    try:
        result = generate_queries_chain.invoke({"topic": topic})
        queries = result.queries if hasattr(result, 'queries') else result['queries']
        print("Generated queries:")
        for q in queries:
            print(f"- {q}")
        return queries
    except Exception as e:
        print(f"Error generating search queries: {e}")
        return [topic]


# ---------- Agent 2: Per-article single key point summarizer ----------

def get_text_from_url(url: str) -> dict:
    """Fetches and extracts text content from a given URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = '\n'.join(chunk for chunk in (phrase.strip() for line in (line.strip() for line in soup.get_text().splitlines()) for phrase in line.split("  ")) if chunk)
        return {"text": text}
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

class OnePointOut(BaseModel):
    key_point: str = Field(description="ONE concise, headline-style bullet point for this article.")

onept_system_prompt = """You are a professional news summarizer. Write EXACTLY ONE objective, headline-style key point in few words.
Rules:
- Be factual and neutral. Prioritize concrete outcomes, numbers, names, places, and dates.
- Use only information in the provided text. Use absolute dates as written; do not normalize.
- No quotes, no emojis, no hashtags, no links, no markdown bullets.
DON'Ts:
- Don't change any NUMBERS. Don't write more than one sentence. Don't include multiple unrelated facts.
- Don't add opinion or analysis. Don't add any extra keys or text outside JSON.
Return ONLY valid JSON: {{"key_point": "<your single concise point>"}}"""

onept_user = "ARTICLE TEXT:```{article_text}```"
onept_prompt = ChatPromptTemplate.from_messages([("system", onept_system_prompt), ("user", onept_user)])
onept_parser = JsonOutputParser(pydantic_object=OnePointOut)
one_point_chain = onept_prompt | llm_json | onept_parser

def summarize_url_to_one_point(url: str) -> dict:
    data = get_text_from_url(url)
    text = (data or {}).get("text", "")
    if not text:
        return {"key_point": f"(No text extracted from {url})"}
    try:
        out = one_point_chain.invoke({"article_text": text})
        return {"key_point": out.key_point if hasattr(out, "key_point") else out["key_point"]}
    except Exception as e:
        return {"key_point": f"(Summarization failed for {url}: {e})"}


# ---------- Agent 3: Final Editor/Merger ----------

class MergeOut(BaseModel):
    summary_points: list[str] = Field(description="Exactly 5 bullets, one per input news, in original order.")

merge_system = """You are an editor. Merge the five single-article key points into EXACTLY FIVE bullets, preserving input order (news1..news5).
Do not re-write or embellish beyond minimal cleanup.
Return ONLY: {{"summary_points": ["...", "...", "...", "...", "..."]}}"""

merge_user = "news1: {n1}\nnews2: {n2}\nnews3: {n3}\nnews4: {n4}\nnews5: {n5}"
merge_prompt = ChatPromptTemplate.from_messages([("system", merge_system), ("user", merge_user)])
merge_parser = JsonOutputParser(pydantic_object=MergeOut)
merge_chain = merge_prompt | llm_json | merge_parser


# ---------- Orchestration Logic ----------

def build_parallel_for_urls(urls: list[str]) -> RunnableParallel:
    return RunnableParallel(**{f"news{i+1}": RunnableLambda(lambda u=u: summarize_url_to_one_point(u)) for i, u in enumerate(urls)})

def summarize_five_urls(urls: list[str]) -> dict:
    if len(urls) != 5:
        raise ValueError("Provide exactly 5 URLs.")
    print("\nExtracting content from 5 URLs and analyzing...")
    
    parallel_summarizer = build_parallel_for_urls(urls)
    individual_summaries = parallel_summarizer.invoke({})

    ordered_points = [individual_summaries[f"news{i+1}"]["key_point"] for i in range(5)]

    print("Compiling final summary...")
    merged = merge_chain.invoke({f"n{i+1}": p for i, p in enumerate(ordered_points)})
    
    final_points = merged.summary_points if hasattr(merged, "summary_points") else merged["summary_points"]
    return {"summary_points": final_points}

def search_for_news(queries: list[str], num_urls: int = 5) -> list[str]:
    """Searches for news articles using a list of queries until enough URLs are found."""
    print("\nSearching for BBC news articles...")
    found_urls = set()
    with DDGS() as ddgs:
        for query in queries:
            if len(found_urls) >= num_urls:
                break
            
            # --- START DEBUG BLOCK ---
            # IMPROVEMENT: Broaden search from 'bbc.com/news' to all of 'bbc.com'
            search_query = f"{query} site:bbc.com"
            print(f"\n[DEBUG] Executing search query: '{search_query}'")
            
            results = ddgs.text(keywords=search_query, region='uk-en', safesearch='off', max_results=num_urls * 2)
            
            print(f"[DEBUG] Raw search returned {len(results)} results.")
            
            for i, r in enumerate(results):
                print(f"[DEBUG]  Result {i+1}:")
                url = r.get('href')
                
                if not url:
                    print("[DEBUG]    -> REJECTED: No 'href' key in result.")
                    continue
                
                print(f"[DEBUG]    URL: {url}")

                is_bbc = 'bbc.com' in url
                if not is_bbc:
                    print("[DEBUG]    -> REJECTED: URL is not from bbc.com.")
                    continue

                is_new = url not in found_urls
                if not is_new:
                    print("[DEBUG]    -> REJECTED: URL is a duplicate.")
                    continue
                
                print("[DEBUG]    -> ACCEPTED: New, valid BBC URL.")
                found_urls.add(url)
                if len(found_urls) >= num_urls:
                    break
            # --- END DEBUG BLOCK ---

    final_urls = list(found_urls)[:num_urls]
    print(f"\nFound {len(final_urls)} unique BBC URLs.")
    return final_urls

# ---------- Main Execution Block ----------
if __name__ == "__main__":
    topic = input("Enter a news topic: ")
    
    # Step 1: Generate multiple search queries
    queries = generate_search_queries(topic)
    
    # Step 2: Use the generated queries to find URLs
    urls = search_for_news(queries, num_urls=5)

    if len(urls) < 5:
        print(f"\nCould not find enough BBC news articles on this topic. Found only {len(urls)}.")
    else:
        # Step 3: Summarize the found URLs
        result = summarize_five_urls(urls)
        print("\n--- FIVE-NEWS SUMMARY (from BBC) ---")
        for p in result["summary_points"]:
            print(f"- {p}")
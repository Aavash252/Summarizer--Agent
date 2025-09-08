# --- imports ---
import os
import time
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_file,write_summary_to_file,get_text_from_url


# --- shared LLM (JSON mode) ---
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# Per-Article Single Key Point
class OnePointOut(BaseModel):
    key_point: str = Field(description="ONE concise, headline-style bullet point for this article.")
onept_system_prompt = "You are a professional news summarizer. Write EXACTLY ONE objective, headline-style key point in few words. Return ONLY valid JSON: {{\"key_point\": \"<your single concise point>\"}}"
onept_user = "ARTICLE TEXT:```{article_text}```"
onept_prompt = ChatPromptTemplate.from_messages([("system", onept_system_prompt), ("user", onept_user)])
onept_parser = JsonOutputParser(pydantic_object=OnePointOut)
one_point_chain = onept_prompt | llm_json | onept_parser

# Per-Article Category
class CategoryOut(BaseModel):
    category: str = Field(description="The primary category of the news article (e.g., 'Sports', 'Politics', 'Technology').")
category_system_prompt = "You are a news classifier. From the article text, identify the single most relevant category (e.g., Sports, Politics, Business, Technology). Return ONLY valid JSON: {{\"category\": \"<your category>\"}}"
category_user = "ARTICLE TEXT:```{article_text}```"
category_prompt = ChatPromptTemplate.from_messages([("system", category_system_prompt), ("user", category_user)])
category_parser = JsonOutputParser(pydantic_object=CategoryOut)
category_chain = category_prompt | llm_json | category_parser

# Per-Article Sentiment
class SentimentOut(BaseModel):
    sentiment: str = Field(description="The overall sentiment of the article (e.g., 'Positive', 'Negative', 'Neutral').")
sentiment_system_prompt = "You are a sentiment analyst. Determine the overall sentiment of the news article (Positive, Negative, or Neutral). Return ONLY valid JSON: {{\"sentiment\": \"<your sentiment>\"}}"
sentiment_user = "ARTICLE TEXT:```{article_text}```"
sentiment_prompt = ChatPromptTemplate.from_messages([("system", sentiment_system_prompt), ("user", sentiment_user)])
sentiment_parser = JsonOutputParser(pydantic_object=SentimentOut)
sentiment_chain = sentiment_prompt | llm_json | sentiment_parser


# --- NEW Generalized Helper to analyze any source ---
def analyze_source(source: str, source_type: str) -> dict:
    """Analyzes a single source (URL or file) and returns its analysis."""
    # print(f"------ Analyzing {source_type} -------")
    if source_type == 'url':
        text = get_text_from_url(source)
    else: 
        text = get_text_from_file(source)

    if not text:
        return {"source": source, "key_point": f"(No text extracted)", "category": "Unknown", "sentiment": "Unknown"}

    try:
        # These are the same agents as before
        analysis_branches = RunnableParallel(key_point=one_point_chain, category=category_chain, sentiment=sentiment_chain)
        result = analysis_branches.invoke({"article_text": text})

        key_point_obj = result["key_point"]
        category_obj = result["category"]
        sentiment_obj = result["sentiment"]

        return {
            "source": source, # Keep track of where the data came from
            "key_point": key_point_obj.key_point if hasattr(key_point_obj, 'key_point') else key_point_obj.get('key_point', 'N/A'),
            "category": category_obj.category if hasattr(category_obj, 'category') else category_obj.get('category', 'N/A'),
            "sentiment": sentiment_obj.sentiment if hasattr(sentiment_obj, 'sentiment') else sentiment_obj.get('sentiment', 'N/A')
        }
    except Exception as e:
        return {"source": source, "key_point": f"(Analysis failed: {e})", "category": "Error", "sentiment": "Error"}


# --- NEW Generalized Orchestration Function ---
def run_analysis(sources: list, source_type: str) -> list:
    """
    Runs the analysis in parallel for any number of sources and returns a list of results.
    """
    if not sources:
        print("No sources to analyze.")
        return []

    print(f"\nStarting analysis for {len(sources)} {source_type}(s)...")
    
    # Build a parallel branch for each source dynamically
    branches = {}
    for i, src in enumerate(sources):
        branches[f"source_{i}"] = RunnableLambda(lambda x, s=src: analyze_source(s, source_type))
    
    parallel_runner = RunnableParallel(**branches)
    
    # Invoke all branches at once
    analysis_results = parallel_runner.invoke({})
    
    # The result is a dictionary like {'source_0': {...}, 'source_1': {...}}.
    # We just want the list of result dictionaries.
    return list(analysis_results.values())





# ---------- Example main ----------
if __name__ == "__main__":
    
    urls = [
    #     "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
    #     "https://www.bbc.com/sport/football/articles/c4gljqwe5g8o",
    #     "https://www.bbc.com/sport/football/articles/crmvygekeyzo",
    #     "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno", 
    ]
    
    news_folder = "news"

    sources_to_process = []
    source_type = ""

    if urls:
        print("URL list is not empty. Processing URLs.")
        sources_to_process = urls
        source_type = "url"
    else:
        print(f"URL list is empty. Looking for .txt files in '{news_folder}' folder.")
        try:
            # Dynamically find all .txt files in the specified folder
            sources_to_process = [
                os.path.join(news_folder, f)
                for f in os.listdir(news_folder)
                if f.endswith('.txt')
            ]
            source_type = "file"
        except FileNotFoundError:
            print(f"Error: The folder '{news_folder}' was not found.")
            sources_to_process = []

    # --- Run the analysis if sources were found ---
    if not sources_to_process:
        print("\nNo articles to process. Please add URLs or create .txt files in the 'news' folder.")
    else:
        start_time = time.time()
        final_results = run_analysis(sources_to_process, source_type)
        end_time = time.time()

        print("\n--- FINAL ANALYSIS REPORT ---")
        for i, p in enumerate(final_results, 1):
            source_display = os.path.basename(p.get('source', 'N/A'))
            print(f"\n--- Article {i} (Source: {source_display}) ---")
            print(f"- Point:     {p.get('key_point')}")
            print(f"- Category:  {p.get('category')}")
            print(f"- Sentiment: {p.get('sentiment')}")

        print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")
        write_summary_to_file(final_results,source_type)
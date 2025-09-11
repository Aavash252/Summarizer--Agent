import os
import time
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_file,write_summary_to_file,get_text_from_url


llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)

class OnePointOut(BaseModel):
    key_point: str = Field(description="ONE concise, headline-style bullet point for this article.")
onept_system_prompt = """You are a professional news summarizer.
        Rules:
        - headline-style bullet points but in few words and shorter lines
        - Be factual and neutral: no opinions, hype, or analysis.
        - Prioritize concrete outcomes, numbers, names, places, and dates.
        - Use only information in the provided text; don't guess or add context.
        - Use absolute dates as written; avoid 'today'/'yesterday'.
        - Preserve proper nouns as written; expand acronyms only if expanded in the text.

        DON'Ts (very important):
        - Don't repeat facts or write near-duplicates; each bullet must cover a distinct facet.
        - Don't add opinion, analysis, recommendations, or hype.
        - Don't use info not in the text; no external context or guesses.
        - Don't change numbers, units, currencies, names, or places; preserve as written.
        - Don't merge multiple unrelated facts into one bullet; one idea per bullet.
        - Don't use quotes, parentheses, emojis, hashtags, links, or markdown bullets.
        - Don't alter capitalization of proper nouns or standardize terminology.
        - Don't add any text outside the JSON; no extra keys, comments, or code fences.
        - Don't include trailing commas or otherwise invalid JSON.
 Return ONLY valid JSON: {{\"key_point\": \"<your single concise point>\"}}"""

onept_user = "ARTICLE TEXT:```{article_text}```"
onept_prompt = ChatPromptTemplate.from_messages([("system", onept_system_prompt), ("user", onept_user)])
onept_parser = JsonOutputParser(pydantic_object=OnePointOut)
one_point_chain = onept_prompt | llm_json | onept_parser

class CategoryOut(BaseModel):
    category: str = Field(description="The primary category of the news article (e.g., 'Sports', 'Politics', 'Technology').")
category_system_prompt = """Assign the best-fitting category for the article.
    Choose one of: Political, Business, Sports, Entertainment, Scientific, General.

    Category definitions & signals:
    - Political: government, elections, public policy, legislation, diplomacy, geopolitics, regulators acting in a policy role.
    - Business: companies, markets, earnings, funding/M&A, corporate strategy, industry competition, jobs/layoffs (as business news).
    - Sports: matches, athletes, leagues, scores, transfers, injuries, adventure sports. EXAMPLES: MOUNTAIN CLIMBING, BIKE RACING , ROCK CLIMBING, CLIMBING EVEREST
    - Entertainment: films/TV/music/gaming/pop culture, celebrities, awards, box office, streaming releases.
    - Scientific: research findings, peer-reviewed studies, experiments, space missions, medicine/biology/physics/CS as science.
    - General: everything.

 Return ONLY valid JSON: {{\"category\": \"<your category>\"}}"""

category_user = "ARTICLE TEXT:```{article_text}```"
category_prompt = ChatPromptTemplate.from_messages([("system", category_system_prompt), ("user", category_user)])
category_parser = JsonOutputParser(pydantic_object=CategoryOut)
category_chain = category_prompt | llm_json | category_parser

class SentimentOut(BaseModel):
    sentiment: str = Field(description="The overall sentiment of the article (e.g., 'Positive', 'Negative', 'Neutral').")
sentiment_system_prompt = """Classify overall sentiment of the article content.
        Return only one of: Positive, Negative, Neutral.
       
 Return ONLY valid JSON: {{\"sentiment\": \"<your sentiment>\"}}"""
sentiment_user = "ARTICLE TEXT:```{article_text}```"
sentiment_prompt = ChatPromptTemplate.from_messages([("system", sentiment_system_prompt), ("user", sentiment_user)])
sentiment_parser = JsonOutputParser(pydantic_object=SentimentOut)
sentiment_chain = sentiment_prompt | llm_json | sentiment_parser


# --- NEW Generalized Helper to analyze any source ---
def analyze_source(source: str, source_type: str) -> dict:
    """Analyzes a single source (URL or file) and returns its analysis."""
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
            "source": source,
            "key_point": key_point_obj.key_point if hasattr(key_point_obj, 'key_point') else key_point_obj.get('key_point', 'N/A'),
            "category": category_obj.category if hasattr(category_obj, 'category') else category_obj.get('category', 'N/A'),
            "sentiment": sentiment_obj.sentiment if hasattr(sentiment_obj, 'sentiment') else sentiment_obj.get('sentiment', 'N/A')
        }
    except Exception as e:
        return {"source": source, "key_point": f"(Analysis failed: {e})", "category": "Error", "sentiment": "Error"}


def run_analysis(sources: list, source_type: str) -> list:
    """
    Runs the analysis in parallel for any number of sources and returns a list of results.
    """
    if not sources:
        print("No sources to analyze.")
        return []
    print(f"\nStarting analysis for {len(sources)} {source_type}(s)...")
    branches = {}
    for i, src in enumerate(sources):
        branches[f"source_{i}"] = RunnableLambda(lambda x, s=src: analyze_source(s, source_type))
    parallel_runner = RunnableParallel(**branches)
    analysis_results = parallel_runner.invoke({})
    unordered_results = list(analysis_results.values())
    source_order = {source_path: i for i, source_path in enumerate(sources)}
    sorted_results = sorted(unordered_results, key=lambda res: source_order[res['source']])
    
    return sorted_results
    
if __name__ == "__main__":
    
    urls = [
        # "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
        # "https://www.bbc.com/sport/football/articles/c4gljqwe5g8o",
        # "https://www.bbc.com/sport/football/articles/crmvygekeyzo",
        # "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno", 
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
            files = sorted(os.listdir(news_folder))
            sources_to_process = [
                os.path.join(news_folder, f)
                for f in files
                if f.endswith('.txt')
            ]
            source_type = "file"
        except FileNotFoundError:
            print(f"Error: The folder '{news_folder}' was not found.")
            sources_to_process = []

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

        print(f"\n Total execution time: {end_time - start_time:.2f} seconds")
        write_summary_to_file(final_results,source_type)
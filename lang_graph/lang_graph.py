# --- imports ---
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
# from tools import get_text_from_url
import requests


# ---------------- Shared LLM ----------------
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0, streaming=True)


def get_text_from_url(url: str) -> str:
    """Fetches the clean text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        publish_date = None

        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            publish_date = time_tag["datetime"]
        elif time_tag:
            publish_date = time_tag.get_text(strip=True)

        if not publish_date:
            meta_date = soup.find("meta", {"property": "article:published_time"})
            if meta_date and meta_date.has_attr("content"):
                publish_date = meta_date["content"]

        if not publish_date:
            meta_date = soup.find("meta", {"name": "date"})
            if meta_date and meta_date.has_attr("content"):
                publish_date = meta_date["content"]
    
        paragraphs = soup.find_all('p')
       
        title = soup.title.string if soup.title else "Untitled"
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return {"title": title,"date": publish_date if publish_date else "Date not found", "text": article_text}
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"



# ---------------- State Schema ----------------
class ArticleState(BaseModel):
    url: str
    text: str = ""
    summary: str = ""
    category: str = ""
    sentiment: str = ""

class SummarizerState(BaseModel):
    articles: list[ArticleState]

# ---------------- Prompts ----------------

# 1. Summarization Prompt
class SummaryOut(BaseModel):
    summary: str = Field(description="ONE concise, headline-style bullet point for this article.")

summary_system = """You are a professional news summarizer.
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
{{"summary": "<your single concise point>"}}"""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system),
    ("user", "ARTICLE TEXT:```{article_text}```"),
])
summary_parser = JsonOutputParser(pydantic_object=SummaryOut)
summary_chain = summary_prompt | llm_json | summary_parser

# 2. Category Classification Prompt
class CategoryOut(BaseModel):
    category: str = Field(description="Main category of the article, e.g., Sports, Politics, Business, Technology, World.")

category_system = """Classify the following article into ONE category.
Return ONLY valid JSON:
{{"category": "<category>"}}"""

category_prompt = ChatPromptTemplate.from_messages([
    ("system", category_system),
    ("user", "ARTICLE TEXT:```{article_text}```"),
])
category_parser = JsonOutputParser(pydantic_object=CategoryOut)
category_chain = category_prompt | llm_json | category_parser

# 3. Sentiment Analysis Prompt
class SentimentOut(BaseModel):
    sentiment: str = Field(description="Sentiment of the article: Positive, Negative, or Neutral.")

sentiment_system = """Analyze the sentiment of the following article.
Return ONLY valid JSON:
{{"sentiment": "<Positive|Negative|Neutral>"}}"""

sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", sentiment_system),
    ("user", "ARTICLE TEXT:```{article_text}```"),
])
sentiment_parser = JsonOutputParser(pydantic_object=SentimentOut)
sentiment_chain = sentiment_prompt | llm_json | sentiment_parser

# ---------------- RunnableParallel Helper ----------------
def process_article(article: ArticleState) -> ArticleState:
    """Process one article: summary + category + sentiment in parallel"""
    parallel = RunnableParallel(
        summary=summary_chain,
        category=category_chain,
        sentiment=sentiment_chain,
    )

    try:
        result = parallel.invoke({"article_text": article.text})

        # ✅ Extract inner values properly
        summary = (
            result["summary"].get("summary")
            if isinstance(result["summary"], dict)
            else result["summary"]
        )
        category = (
            result["category"].get("category")
            if isinstance(result["category"], dict)
            else result["category"]
        )
        sentiment = (
            result["sentiment"].get("sentiment")
            if isinstance(result["sentiment"], dict)
            else result["sentiment"]
        )
    except Exception as e:
        summary, category, sentiment = f"(Error: {e})", "Unknown", "Neutral"

    return ArticleState(
        url=article.url,
        text=article.text,
        summary=summary or "(No summary)",
        category=category or "Unknown",
        sentiment=sentiment or "Neutral",
    )


# ---------------- LangGraph Nodes ----------------
def fetch_node(state: SummarizerState):
    """Fetch article text for all URLs"""
    updated_articles = []
    for article in state.articles:
        data = get_text_from_url(article.url)
        text = (data or {}).get("text", "")
        updated_articles.append(ArticleState(url=article.url, text=text))
    return {"articles": updated_articles}

def process_node(state: SummarizerState):
    """Process all articles in parallel"""
    updated_articles = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_article, state.articles))
        updated_articles.extend(results)

    # Print each article as soon as it's done
    for idx, art in enumerate(updated_articles, start=1):
        print(f"\n Article {idx}/{len(updated_articles)} processed:")
        print(f"URL      : {art.url}")
        print(f"Summary  : {art.summary}")
        print(f"Category : {art.category}")
        print(f"Sentiment: {art.sentiment}")

    return {"articles": updated_articles}

def merge_node(state: SummarizerState):
    """Return final structured JSON"""
    return {
        "articles": [
            {
                "url": art.url,
                "summary": art.summary,
                "category": art.category,
                "sentiment": art.sentiment,
            }
            for art in state.articles
        ]
    }

# ---------------- Build LangGraph ----------------
graph = StateGraph(SummarizerState)

graph.add_node("fetch", fetch_node)
graph.add_node("process", process_node)
# graph.add_node("merge", merge_node)

graph.set_entry_point("fetch")
graph.add_edge("fetch", "process")
# graph.add_edge("process", "merge")
graph.add_edge("process", END)

app = graph.compile()

# ---------------- Orchestration ----------------
def summarize_articles(urls: list[str]) -> dict:
    initial_state = {"articles": [ArticleState(url=u) for u in urls]}
    result = app.invoke(initial_state)
    return result

# ---------------- Example Main ----------------
if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
        "https://www.bbc.com/sport/football/articles/c4gljqwe5g8o",
        "https://www.bbc.com/sport/football/articles/crmvygekeyzo",
        "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno",
    ]

    start_time = time.time()
    result = summarize_articles(urls)
    end_time = time.time()

    # print("\n========== FINAL NEWS SUMMARY ==========")
    # for article in result["articles"]:
    #     print(f"\nURL      : {article['url']}")
    #     print(f"Summary  : {article['summary']}")
    #     print(f"Category : {article['category']}")
    #     print(f"Sentiment: {article['sentiment']}")

    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")

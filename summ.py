# --- imports ---
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_url
from datetime import datetime

# --- shared LLM (you can also use different models per agent) ---
llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# ---------- Agent 1: Summarizer ----------
class SummaryOut(BaseModel):
    summary_points: list[str] = Field(description="Exactly 3 concise bullets.")

summ_system_prompt = """You are a professional news summarizer.
Write EXACTLY 3 objective, .
Rules:
- headline-style bullet points but in few words and shorter lines
- Be factual and neutral: no opinions, hype, or analysis.
- Prioritize concrete outcomes, numbers, names, places, and dates.
- Use only information in the provided text; don't guess or add context.
- Use absolute dates as written; avoid 'today'/'yesterday'.
- Preserve proper nouns as written; expand acronyms only if expanded in the text.

DON'Ts (very important):
- Don't produce more or fewer than 3 bullets.
- Don't repeat facts or write near-duplicates; each bullet must cover a distinct facet.
- Don't add opinion, analysis, recommendations, or hype.
- Don't use info not in the text; no external context or guesses.
- Don't change numbers, units, currencies, names, or places; preserve as written.
- Don't merge multiple unrelated facts into one bullet; one idea per bullet.
- Don't use quotes, parentheses, emojis, hashtags, links, or markdown bullets.
- Don't alter capitalization of proper nouns or standardize terminology.
- Don't add any text outside the JSON; no extra keys, comments, or code fences.
- Don't include trailing commas or otherwise invalid JSON.


Return ONLY a valid JSON object with keys: 
- "summary_points": list of 3 bullet points  """
sum_user = "ARTICLE TEXT:```{article_text}```"

sum_prompt = ChatPromptTemplate.from_messages([
    ("system", summ_system_prompt),
    ("user", sum_user),
])

sum_parser = JsonOutputParser(pydantic_object=SummaryOut)
summarizer_chain = sum_prompt | llm_json | sum_parser

# ---------- Agent 2: Sentiment ----------
class SentimentOut(BaseModel):
    sentiment: str = Field(description="One of: Positive, Negative, Neutral")

sent_system_prompt = """Classify overall sentiment of the article content.
Return only one of: Positive, Negative, Neutral.
Labels:
- positive: concrete good outcomes or clearly supportive tone. Also if someone achives something than thats POSITIVE
- negative: concrete harm, risk, sanctions, losses, or clearly critical tone.
- neutral: factual/mixed/uncertain (rumors, forecasts, unresolved investigations).

DON'Ts (very important):
    - Don't use labels other than "Positive", "Negative", or "Neutral".
    - Don't use info outside the provided text. No external knowledge or assumptions.
    - Don't apply moral judgments.

Return ONLY a valid JSON object with keys: 

"sentiment": one of the three sentiment labels
 """
sent_user = "ARTICLE TEXT:```{article_text}```"

sent_prompt = ChatPromptTemplate.from_messages([
    ("system", sent_system_prompt),
    ("user", sent_user),
])

sent_parser = JsonOutputParser(pydantic_object=SentimentOut)
sentiment_chain = sent_prompt | llm_json | sent_parser

# ---------- Agent 3: Category ----------
class CategoryOut(BaseModel):
    category: str = Field(description='One of: Political, Business, Sports, Entertainment, Scientific, General')

cate_system_prompt = """Assign the best-fitting category for the article.
Choose one of: Political, Business, Sports, Entertainment, Scientific, General.

Category definitions & signals:
- Political: government, elections, public policy, legislation, diplomacy, geopolitics, regulators acting in a policy role.
- Business: companies, markets, earnings, funding/M&A, corporate strategy, industry competition, jobs/layoffs (as business news).
- Sports: matches, athletes, leagues, scores, transfers, injuries, adventure sports. EXAMPLES: MOUNTAIN CLIMBING, BIKE RACING , ROCK CLIMBING, CLIMBING EVEREST
- Entertainment: films/TV/music/gaming/pop culture, celebrities, awards, box office, streaming releases.
- Scientific: research findings, peer-reviewed studies, experiments, space missions, medicine/biology/physics/CS as science.
- General: everything.

Return ONLY a valid JSON object with keys: 
"category": one of the labels of category """
cat_user = "ARTICLE TEXT:```{article_text}```"

cat_prompt = ChatPromptTemplate.from_messages([
    ("system", cate_system_prompt),
    ("user", cat_user),
])

cat_parser = JsonOutputParser(pydantic_object=CategoryOut)
category_chain = cat_prompt | llm_json | cat_parser

# ---------- Optional: Editor/Merger Agent ----------

class EditorOut(BaseModel):
    summary_points: list[str]
    sentiment: str
    category: str

editor_sys = """You are an editor who merges specialist agents' outputs into one clean JSON.
- Keep summary_points exactly as provided.
- Ensure sentiment is one of: Positive, Negative, Neutral.
- Ensure category is one of: Political, Business, Sports, Entertainment, Scientific, General.
Return only JSON with keys: summary_points, sentiment, category."""
editor_user = """
TITLE: {title}
PUBLISHED: {published}
SPECIALIST_OUTPUTS:
- SUMMARY: {summary_json}
- SENTIMENT: {sentiment_json}
- CATEGORY: {category_json}
"""

editor_prompt = ChatPromptTemplate.from_messages([
    ("system", editor_sys),
    ("user", editor_user),
])

editor_parser = JsonOutputParser(pydantic_object=EditorOut)
editor_chain = editor_prompt | llm_json | editor_parser

# ---------- Orchestration ----------
# Run three specialists in parallel, then (optionally) send to editor.
def analyze_article(article_text: str, title: str = "", published: str = ""):
    parallel = RunnableParallel(
        summary=summarizer_chain,
        sentiment=sentiment_chain,
        category=category_chain
    )
    print("######")
    tri_output = parallel.invoke({"article_text": article_text})
    # print(tri_output)
    # sentiment = tri_output.get('sentiment', 'N/A')
    # category = tri_output.get('category', 'N/A')
    # print("######",sentiment,category)
    

    # If you want raw agent outputs (no editor):
    return {
    "summary_points": tri_output["summary"]["summary_points"],
    "sentiment": tri_output["sentiment"]["sentiment"],
    "category": tri_output["category"]["category"],
}

    # Send to Editor for final polish/validation:
    # merged = editor_chain.invoke({
    #     "title": title or "Untitled",
    #     "published": published or "Unknown",
    #     "summary_json": tri_output["summary"].model_dump_json(),
    #     "sentiment_json": tri_output["sentiment"].model_dump_json(),
    #     "category_json": tri_output["category"].model_dump_json(),
    # })
    
    # return merged.model_dump()

def format_publish_date(iso_date: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error formatting date: {e}"
    
if __name__ == "__main__":
    target_url = "https://www.bbc.com/news/articles/cm2y70xknnyo"
    print(f"-> Fetching article content from: {target_url}")
    
    article_data = get_text_from_url(target_url)
    title = article_data["title"]
    article_text = article_data["text"]
    date = article_data["date"]

    if not article_text:
        print("-> Could not retrieve article text. Exiting.")
        exit()

    frt_date = format_publish_date(date)
    print("-> Article content fetched successfully. Analyzing...")

    try:
        result = analyze_article(article_text, title, frt_date)
        #print (result)

        print("\n-------------------")
        print("Analysis Complete:")
        print(f"TITLE: {title}")
        print(f"Published Date: {frt_date}")
        print(f"CATEGORY: {result['category']}")

        print("\nSUMMARY:")
        for point in result["summary_points"]:
            print(f"- {point}")

        print(f"\nSENTIMENT: {result['sentiment']}")
        print("-------------------\n")

    except Exception as e:
        print(f"An error occurred: {e}")

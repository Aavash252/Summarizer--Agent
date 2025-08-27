from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_url # Assuming 'tools.py' contains get_text_from_url
from datetime import datetime
from typing import List

# --- 1. Modified Pydantic Model for Multiple Article Summaries ---
class MultiArticleSummary(BaseModel):
    # This will hold a list of summary points, one for each article
    overall_summary_points: List[str] = Field(description="A list of EXACTLY 4 concise, objective bullet points, one for each article.")
    # overall_sentiment: str = Field(description="The overall sentiment across all articles: 'Positive', 'Negative', or 'Neutral'.")
    # overall_category: str = Field(description="The dominant category across all articles, e.g., 'Political', 'Business', 'Sports', 'Entertainment', 'Scientific', 'General'.")

# --- 2. Set up the LLM with JSON Mode (No change needed) ---
llm = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# --- 3. Adjusted System Prompt for Multiple Articles ---
# system_prompt_multi = """You are a professional news journalist. 
# Your task is to carefully read the provided article texts 
# Present these as a list of EXACTLY 4 concise bullet points.
# ONE POINT FOR EACH NEWS ARTICLE 

# Do :
# For EACH of the 4 provided articles:

# Rules:
# - headline-style bullet points but in few words and shorter lines
# - Be factual and neutral: no opinions, hype, or analysis.
# - Prioritize concrete outcomes, numbers, names, places, and dates.
# - Use only information in the provided text; don't guess or add context.
# - Use absolute dates as written; avoid 'today'/'yesterday'.
# - Preserve proper nouns as written; expand acronyms only if expanded in the text.

# DON'Ts (very important):
# - Don't produce more or fewer than 3 bullets.
# - Don't repeat facts or write near-duplicates; each bullet must cover a distinct facet.
# - Don't add opinion, analysis, recommendations, or hype.
# - Don't use info not in the text; no external context or guesses.
# - Don't change numbers, units, currencies, names, or places; preserve as written.
# - Don't merge multiple unrelated facts into one bullet; one idea per bullet.
# - Don't use quotes, parentheses, emojis, hashtags, links, or markdown bullets.
# - Don't alter capitalization of proper nouns or standardize terminology.
# - Don't add any text outside the JSON; no extra keys, comments, or code fences.
# - Don't include trailing commas or otherwise invalid JSON.
   
   
# Return ONLY a valid JSON object with: 
# - "summary_points": list of 4 bullet points (one per article)

# """
system_prompt_multi = """You are a professional news journalist. 
Your task is to carefully read the 4 provided news articles.

For each of the 4 articles, you must extract its single most important key fact or development.
Consolidate these into a list of EXACTLY 4 concise, factual bullet points. Each bullet point MUST correspond to one article.

Rules for each bullet point:
- Be a very short, headline-style summary.
- Be strictly factual and neutral; no opinions, hype, or analysis.
- Prioritize concrete outcomes, numbers, names, places, and dates.
- Use only information explicitly present in the provided article text.
- Avoid absolute dates like 'today' or 'yesterday' unless directly quoted; use dates as written.
- Preserve proper nouns as written; do not expand acronyms unless the full form is in the text.

Return ONLY a valid JSON object. The JSON object must have a single key:
"overall_summary_points": [
    "Summary for Article 1 (very short)",
    "Summary for Article 2 (very short)",
    "Summary for Article 3 (very short)",
    "Summary for Article 4 (very short)"
]

Ensure the output is valid JSON and contains no other text or keys.
"""

# The user prompt template needs to accept multiple article texts.
# We'll concatenate them with clear separators for the LLM.
user_prompt_multi = """Here are 4 news articles for you to analyze:

ARTICLE 1:
```{article_text_1}```

ARTICLE 2:
```{article_text_2}```

ARTICLE 3:
```{article_text_3}```

ARTICLE 4:
```{article_text_4}```
"""

# The prompt creation for the multi-article summary.
prompt_multi = ChatPromptTemplate.from_messages([
    ("system", system_prompt_multi),
    ("user", user_prompt_multi)
])

# --- 4. Set up the Parser and Build the Chain for Multiple Articles ---
parser_multi = JsonOutputParser(pydantic_object=MultiArticleSummary)
chain_multi = prompt_multi | llm | parser_multi

def format_publish_date(iso_date: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error formatting date: {e}"

# --- 5. Main Function to Summarize and Print Multiple Articles ---
def summarize_multiple_articles(urls: List[str]):
    """
    Fetches articles from a list of URLs, summarizes them into 4 points,
    and prints the consolidated result.
    """
    if len(urls) != 4:
        print("Please provide exactly 4 URLs for this function.")
        return

    article_texts = []
    article_titles = []
    article_dates = []

    print("-> Fetching content for 4 articles...")
    for i, url in enumerate(urls):
        print(f"-> Fetching article {i+1} from: {url}")
        article_data = get_text_from_url(url)
        if article_data and article_data["text"]:
            article_texts.append(article_data["text"])
            article_titles.append(article_data["title"])
            article_dates.append(article_data["date"])
        else:
            print(f"-> Could not retrieve article text for {url}. Skipping.")
            article_texts.append("") # Add empty string to maintain count
            # article_titles.append("N/A")
            # article_dates.append("N/A")

    if not all(article_texts):
        print("-> One or more articles could not be fetched. Exiting.")
        return

    print("-> All article contents fetched successfully. Analyzing...")

    try:
        # Prepare the input dictionary for the chain
        input_data = {
            "article_text_1": article_texts[0],
            "article_text_2": article_texts[1],
            "article_text_3": article_texts[2],
            "article_text_4": article_texts[3],
        }
       
        result = chain_multi.invoke(input_data)
        print(result)
        
        print("\n-------------------")
        print("Consolidated Analysis Complete:")
        
        print("\nIndividual Article Information:")
        # for i in range(4):
        #     print(f"  Article {i+1} Title: {article_titles[i]}")
        #     frt_date = format_publish_date(article_dates[i])
        #     print(f"  Published Date: {frt_date}")
        #     print("-" * 20)

        # overall_category = result.get('overall_category', 'N/A')
        # print(f"\nOVERALL CATEGORY: {overall_category}")

        print("\nCONSOLIDATED 4-POINT SUMMARY:")
        overall_summary_points = result.get('overall_summary_points', [])
        print(overall_summary_points)
        
        if overall_summary_points:
            for i, point in enumerate(overall_summary_points):
                print(f"- Article {i+1}: {point}")
        else:
            print("Could not generate consolidated summary points.")
        
        # overall_sentiment = result.get('overall_sentiment', 'N/A')
        # print(f"\nOVERALL SENTIMENT: {overall_sentiment}")
        print("-------------------\n")

    except Exception as e:
        print(f"An error occurred during multi-article summarization: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    target_urls = [
        "https://www.bbc.com/sport/football/live/ce93m7rrzzvt",
        "https://www.bbc.com/sport/football/live/cn5epw6vqyzt",
        "https://www.bbc.com/sport/football/articles/cgjye1144l2o",
        "https://www.bbc.com/sport/football/articles/cjeyjwq9kkno",     # Example science article
    ]
    summarize_multiple_articles(target_urls)
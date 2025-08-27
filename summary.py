
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from tools import get_text_from_url
from datetime import datetime


class ArticleSummary(BaseModel):
    summary_points: list[str] = Field(description="A list of EXACTLY 3 concise, objective bullet points summarizing the article.")
    sentiment: str = Field(description="The overall sentiment of the article, which must be one of the following: 'Positive', 'Negative', or 'Neutral'.")

# --- 2. Set up the LLM with JSON Mode (No change needed) ---
llm = ChatOllama(model="qwen3:4b", format="json", temperature=0)


system_prompt = """You are a professional news journalist. 
Your task is to carefully read the provided article text and produce a clear, objective news brief.

Do three things:
1. Write EXACTLY 3 concise bullet points that capture the key facts and developments of the article, in the style of news headlines.
   - Avoid opinions or analysis — focus only on what is stated in the article.
2. Assign the overall sentiment of the article: "Positive", "Negative", or "Neutral".
   - If sentiment is mixed or unclear, classify it as "Neutral".
3. Assign a category to the news like "Political", "Business", "Sports", "Entertainment","Scientific".
    - If unclear just mark it as "General"


Return ONLY a valid JSON object with three keys: 
- "summary_points": list of 3 bullet points
- "sentiment": one of the three sentiment labels
- "category": one of the labels of category 
"""

# The user prompt template remains the same.
user_prompt = """ARTICLE TEXT:```{article_text}```"""

# The prompt creation remains the same.
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])


# --- 4. Set up the Parser and Build the Chain (No change needed) ---
parser = JsonOutputParser(pydantic_object=ArticleSummary)
chain = prompt | llm | parser

# --- 5. Main Function to Summarize and Print ---
def summarize_article(url: str):
    """
    Fetches an article from a URL, summarizes it, and prints the result.
    """
    print(f"-> Fetching article content from: {url}")
    
    article_data = get_text_from_url(url)
    title = article_data["title"]
    article_text = article_data["text"]
    date = article_data["date"]
    
    if not article_text:
        print("-> Could not retrieve article text. Exiting.")
        return

    print("-> Article content fetched successfully. Analyzing...")
    
    try:
        result = chain.invoke({"article_text": article_text})
        
        # --- 3. UPDATE THE PRINTED OUTPUT --- (_HERE_ 3/3)
        # We now print the sentiment along with the summary points.
        print("\n-------------------")
        print("Analysis Complete:")
        
        # Access and print the sentiment from the result dictionary
        print(f"TITLE: {title}")
        frt_date= format_publish_date(date)
        print(f"Published Date: {frt_date}")
        sentiment = result.get('sentiment', 'N/A')
        category = result.get('category', 'N/A')
        
        print(f"CATEGORY: {category}")

        print("\nSUMMARY:")
       
        summary_points = result.get('summary_points', [])
        
        if summary_points:
            for point in summary_points:
                print(f"- {point}")
        else:
            print("Could not generate summary points.")
        
        print(f"\nSENTIMENT: {sentiment}")
        print("-------------------\n")

    except Exception as e:
        print(f"An error occurred during summarization: {e}")

def format_publish_date(iso_date: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error formatting date: {e}"
    
# --- Main execution block (No change needed) ---
if __name__ == "__main__":
    target_url = "https://www.bbc.com/sport/football/articles/cx2q009xzgeo" 
    summarize_article(target_url)
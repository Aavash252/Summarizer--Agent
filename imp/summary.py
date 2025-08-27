# main.py

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
# Import the function from our new tool file
from tools import get_text_from_url

# --- 1. Define the Desired JSON Structure ---
class ArticleSummary(BaseModel):
    summary_points: list[str] = Field(description="A list of EXACTLY 4 concise, objective bullet points summarizing the article.")

# --- 2. Set up the LLM with JSON Mode ---
llm = ChatOllama(model="qwen3:4b", format="json", temperature=0)

# --- 3. Create a Strict Prompt for Bullet Points ---
prompt = ChatPromptTemplate.from_template(
    """You are a highly skilled news summarizer. Your only job is to analyze the provided article text
    and generate 3 concise, objective bullet points.
    Return ONLY a valid JSON object with a single key "summary_points" which contains the list of these 3 bullet points.

    ARTICLE TEXT:
    ```{article_text}```
    """
)

# --- 4. Set up the Parser and Build the Chain ---
parser = JsonOutputParser(pydantic_object=ArticleSummary)
chain = prompt | llm | parser

# --- 5. Main Function to Summarize and Print ---
def summarize_article(url: str):
    """
    Fetches an article from a URL, summarizes it, and prints the result.
    """
    print(f"-> Fetching article content from: {url}")
    
    # Step 1: Use our tool to get the article text
    article_text = get_text_from_url(url)
    
    # If the tool failed, stop here.
    if not article_text:
        print("-> Could not retrieve article text. Exiting.")
        return

    print("-> Article content fetched successfully. Analyzing...")
    
    try:
        # Step 2: Get the structured summary from the chain
        result = chain.invoke({"article_text": article_text})
        
        # Step 3: Format the output in a clean, readable way
        print("\n-------------------")
        print("Here are 3 concise, objective bullet points summarizing the article:\n")
        
        summary_points = result.get('summary_points', [])
        
        if summary_points:
            for point in summary_points:
                print(f"- {point}")
        else:
            print("Could not generate summary points.")
            
        print("-------------------\n")

    except Exception as e:
        print(f"An error occurred during summarization: {e}")

if __name__ == "__main__":
    target_url = "https://www.bbc.com/news/articles/cm2y70xknnyo" 
    summarize_article(target_url)
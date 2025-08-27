
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from json_tool import get_article_data 

# --- 1. Define the Desired JSON Structure ---
# This class defines the schema for our output.
class Summary(BaseModel):
    Title: str = Field(description="The main title of the article.")
    summary_points: list[str] = Field(description="A list of EXACTLY 3 bullet points summarizing the article.")
    sentiment: str = Field(description="The overall sentiment of the article, must be one of: 'Negative','Positive', or 'Neutral'.")


llm = ChatOllama(model="qwen3:4b" ,format="json", temperature=0)

json_prompt = ChatPromptTemplate.from_template(
    """You are an expert news analyst who provides structured JSON output.
    Analyze the following article text and provide a structured summary.
    Follow the formatting instructions precisely.

    FORMATTING_INSTRUCTIONS:
    {format_instructions}

    ARTICLE_TEXT:
    {article_text}
    """
)

json_parser = JsonOutputParser(pydantic_object=Summary)



# --- 4. Build the Chain ---
# We link the prompt, the model, and the JSON parser together.
json_chain = json_prompt | llm | json_parser

# --- 5. Main Execution Function ---
def run_json_summarizer(url):
    """Fetches article, runs the JSON chain, and prints the result."""
    print(f"Attempting to summarize URL: {url}")
    
    # Step 1: Fetch article data using our tool
    # article_data = get_article_data(url)
    article_data = input("enter the news")
    
    # Handle potential errors from the tool
    if "error" in article_data:
        print(f"Error: {article_data['error']}")
        return

    article_text = article_data["text"]
    print("Article content fetched successfully.")
    print("Invoking LLM chain for structured summary...")

    try:
        # Step 2: Invoke the chain with the required inputs
        # The parser's format instructions are crucial for the LLM to know the schema.
        result = json_chain.invoke({
            "article_text": article_text,
            "format_instructions": json_parser.get_format_instructions()
        })
        
        # Step 3: Print the structured result
        print("\n----- STRUCTURED SUMMARY (JSON) -----")
        # 'result' is now a Python dictionary, we can print it nicely
        print(json.dumps(result, indent=2))
        print("-------------------------------------\n")
        
        # You can now easily access specific parts of the summary
        print("Accessing a specific field: ")
        print(f"Sentiment: {result['sentiment']}")

    except Exception as e:
        print(f"An error occurred while running the chain: {e}")

# --- Main execution block ---
# if __name__ == "__main__":
#     # You can replace this URL with any news article
#     # target_url = "https://www.bbc.com/news/articles/cm2y70xknnyo"
#     # run_json_summarizer(target_url)
#     text= input("enter the news")




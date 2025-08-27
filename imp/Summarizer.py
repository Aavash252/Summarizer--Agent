# 1. Import necessary components from LangChain and our tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from tools import get_text_from_url

# 2. Connect to the local LLM (Ollama)
# Make sure Ollama is running in the background!
llm = ChatOllama(model="qwen3:4b",format="json", temperature=0)

# 3. Create the Prompt Template (This is Prompt Engineering!)
# We create a template that has a system message to set the context
# and a human message that will contain our variable (the article text).
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert news analyst. Your job is to provide clear, concise, and objective summaries of news articles. Present just 3 bullet points"),
    ("human", "Please summarize the following article:\n\n{article_text}")
])

# 4. Create the Output Parser
# This will simply take the LLM's output and convert it into a simple string.
output_parser = StrOutputParser()

# 5. Build the LangChain "Chain"
# We use the | (pipe) operator to link the components together.
# This is called LangChain Expression Language (LCEL).
# The flow is: prompt -> llm -> output_parser
chain = prompt | llm | output_parser

# 6. Run the summarizer
def summarize_article(url):
    """Fetches an article from a URL and summarizes it using the LangChain chain."""
    print("Fetching article content...")
    article_content = get_text_from_url(url)
    
    if "Error" in article_content:
        print(article_content)
        return

    print("Article content fetched. Summarizing...")
    
    # "Invoke" the chain with the article content.
    # The dictionary key 'article_text' must match the variable in our prompt.
    summary = chain.invoke({"article_text": article_content})
    
    print("\n----- SUMMARY -----")
    # print(summary)
    summary_points = summary.get('summary', [])
        
    if summary_points:
        for point in summary_points:
            print(f"- {point}")
    else:
        print("Could not generate summary points.")
    print("-------------------\n")

# Main execution block
if __name__ == "__main__":
    # Replace this with any news article URL you want to summarize
    target_url = "https://www.bbc.com/news/articles/cm2y70xknnyo" 
    # target_input = input("Please Enter the target URL")
    summarize_article(target_url)
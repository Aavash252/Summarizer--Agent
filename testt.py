import re
import json
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from tools import get_text_from_file,get_text_from_url

class GraphState(TypedDict):
    source_text: str   
    key_points: dict  
    summary_text: str 
    final_output: dict 

class KeyPoints(BaseModel):
    who: str = Field(description="Identify the key people, groups, or entities involved.")
    what: str = Field(description="Describe the main event or topic of the article.")
    when: str = Field(description="Specify the date, time, or timeframe of the events.")
    where: str = Field(description="Pinpoint the location or setting of the events.")
    why: str = Field(description="Explain the reason, motivation, or cause behind the events.")

class Sentiment(BaseModel):
    sentiment: str = Field(description="The overall sentiment of the article (e.g., 'Positive', 'Negative', 'Neutral').")
sentiment_system_prompt = """Classify overall sentiment of the article content.
        Return only one of: Positive, Negative, Neutral.
       
 Return ONLY valid JSON: {{\"sentiment\": \"<your sentiment>\"}}"""
class Category(BaseModel):
    category: str = Field(description="The primary category of the news article (e.g., 'Sports', 'Politics', 'Technology', 'Business').")
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

class FinalReport(BaseModel):
    summary: str = Field(description="The final, human-readable summary of the article.")
    sentiment: str = Field(description="The overall sentiment of the article.")
    category: str = Field(description="The primary category of the news article.")

llm = ChatOllama(model="qwen3:4b", temperature=0)

extractor_prompt = ChatPromptTemplate.from_template("Extract the key '5 Ws' from the following article. Respond with ONLY valid JSON.\n\nARTICLE:\n```{source_text}```")
extractor_chain = extractor_prompt | llm.bind(format="json") | JsonOutputParser(pydantic_object=KeyPoints)

summarizer_prompt = ChatPromptTemplate.from_template("Synthesize the following key points into a single, well-written summary.Just one line. Use ONLY the information provided.\n\nKEY POINTS:\n{key_points}")
summarizer_chain = summarizer_prompt | llm | StrOutputParser()

# sentiment_prompt = ChatPromptTemplate.from_template("Analyze the sentiment of the following article. Respond with ONLY valid JSON.\n\nARTICLE:\n```{source_text}```")
# sentiment_chain = sentiment_prompt | llm.bind(format="json") | JsonOutputParser(pydantic_object=Sentiment)


sentiment_user = "ARTICLE TEXT:```{source_text}```"
sentiment_prompt = ChatPromptTemplate.from_messages([("system", sentiment_system_prompt), ("user", sentiment_user)])
sentiment_parser = JsonOutputParser(pydantic_object=Sentiment)
sentiment_chain = sentiment_prompt | llm.bind(format="json") | sentiment_parser

category_user = "ARTICLE TEXT:```{source_text}```"
category_prompt = ChatPromptTemplate.from_messages([("system", category_system_prompt), ("user", category_user)])
category_parser = JsonOutputParser(pydantic_object=Category)
category_chain = category_prompt | llm.bind(format="json") | category_parser
# category_chain = category_prompt | llm.bind(format="json") | JsonOutputParser(pydantic_object=Category)

def key_point_extractor_node(state: GraphState) -> dict:
    """Agent 1: Extracts structured key points (5 Ws)."""
    print("---  Agent 1: Extracting Key Points ---")
    key_points_json = extractor_chain.invoke({"source_text": state["source_text"]})
    return {"key_points": key_points_json}

def prose_summarizer_node(state: GraphState) -> dict:
    """Agent 2: Writes a prose summary from key points."""
    print("---  Agent 2: Writing Prose Summary ---")
    summary_text = summarizer_chain.invoke({"key_points": state["key_points"]})
    clean_draft = clean_llm_output(summary_text)

    return {"summary_text": clean_draft}

def analysis_and_formatting_node(state: GraphState) -> dict:
    """Agent 3: Analyzes sentiment/category and formats the final report."""
    print("---  Agent 3: Analyzing and Formatting ---")
    
    analysis_runner = RunnableParallel(
        sentiment=sentiment_chain,
        category=category_chain
    )
    
    analysis_results = analysis_runner.invoke({"source_text": state["source_text"]})
    
    final_report = FinalReport(
        summary=state['summary_text'],
        sentiment=analysis_results['sentiment']['sentiment'],
        category=analysis_results['category']['category']
    )
    
    return {"final_output": final_report.model_dump()}

workflow = StateGraph(GraphState)

workflow.add_node("extractor", key_point_extractor_node)
workflow.add_node("summarizer", prose_summarizer_node)
workflow.add_node("analyzer", analysis_and_formatting_node)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "summarizer")
workflow.add_edge("summarizer", "analyzer")
workflow.add_edge("analyzer", END)

app = workflow.compile()

def clean_llm_output(raw_text: str) -> str:
    """
    Removes unwanted <think> tags and their content from the LLM's output.
    """
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

if __name__ == "__main__":
    print("Welcome to the Structured Summarizer Agent!")

    source_input= "news/news1.txt"
    
    if source_input.startswith("http"):
        text = get_text_from_url(source_input)
    else:
        text = get_text_from_file(source_input)

    if text.startswith("Error:"):
        print(text)
    else:
        initial_state = {"source_text": text}
        final_result = app.invoke(initial_state)

        print("\n\n--- Key Points ---")
        for key, value in final_result['key_points'].items():
            print(f"- {key.capitalize()}: {value}")

        report = final_result['final_output']
        print("\n--- RESULTS ---")
        print(f"\n- Summary:",report.get('summary', 'N/A')) 
        print(f"- Category:",report.get('category', 'N/A'))
        print(f"- Sentiment:",report.get('sentiment', 'N/A'))
        
    
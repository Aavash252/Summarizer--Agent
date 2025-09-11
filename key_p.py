import re
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from tools import get_text_from_file,write_summary_to_file,get_text_from_url

class GraphState(TypedDict):
    source_text: str   
    key_points: dict   
    final_summary: str 


# llm_json = ChatOllama(model="qwen3:4b", format="json", temperature=0)
# llm_text = ChatOllama(model="qwen3:4b", temperature=0)

llm = ChatOllama(model="qwen3:4b", temperature=0)


class KeyPoints(BaseModel):
    who: str = Field(description="Identify the key people, groups, or entities involved.")
    what: str = Field(description="Describe the main event or topic of the article.")
    when: str = Field(description="Specify the date, time, or timeframe of the events.")
    where: str = Field(description="Pinpoint the location or setting of the events.")
    why: str = Field(description="Explain the reason, motivation, or cause behind the events.")

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting key information from news articles. Your goal is to identify the '5 Ws': Who, What, When, Where, and Why."),
    ("user", "Please extract these key points from the following article text. Respond with ONLY a valid JSON object.\n\nARTICLE:\n```{source_text}```")
])

json_parser = JsonOutputParser(pydantic_object=KeyPoints)
extractor_chain = extractor_prompt | llm.bind(format="json") | json_parser


summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert news writer. Your task is to write a concise, professional summary paragraph."),
    ("user", """Please synthesize the following key points into a single, well-written summary paragraph.
IMPORTANT: Use ONLY the information provided in these key points. Do not add any outside information or details from the original source.

KEY POINTS:
Who: {who}
What: {what}
When: {when}
Where: {where}
Why: {why}
""")
])
summarizer_chain = summarizer_prompt | llm | StrOutputParser()

def clean_llm_output(raw_text: str) -> str:
    """
    Removes unwanted <think> tags and their content from the LLM's output.
    """
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()


def key_point_extractor_node(state: GraphState) -> dict:
    """
    Extracts structured key points (5 Ws) from the source text.
    """
    print("--- Agent 1: Extracting Key Points ---")
    source_text = state["source_text"]
    key_points_json = extractor_chain.invoke({"source_text": source_text})
    print("=== Extraction Complete ===")
    return {"key_points": key_points_json}

def prose_summarizer_node(state: GraphState) -> dict:
    """
    Writes a final summary using only the extracted key points.
    """
    print("\n --- Agent 2: Writing Final Summary ---")
    key_points = state["key_points"]
    final_summary = summarizer_chain.invoke(key_points)
    clean_draft = clean_llm_output(final_summary)

    print("--- Summary Complete ---")
    return {"final_summary": clean_draft}

workflow = StateGraph(GraphState)
workflow.add_node("extractor", key_point_extractor_node)
workflow.add_node("summarizer", prose_summarizer_node)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "summarizer")
workflow.add_edge("summarizer", END)

app = workflow.compile()
if __name__ == "__main__":
    print("Welcome to the Key-Point Extractor Summarizer!")

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
        print("\n--- EXTRACTED KEY POINTS ---")
        for key, value in final_result['key_points'].items():
            print(f"- {key.capitalize()}: {value}")
        print("\n--- FINAL SUMMARY ---")
        print(final_result['final_summary'])
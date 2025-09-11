import os
import re
from typing import TypedDict
from tools import get_text_from_file,write_summary_to_file,get_text_from_url
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
class GraphState(TypedDict):
    source: str          # The file path or URL
    original_text: str
    current_summary: str
    user_feedback: str
    revision_number: int

llm = ChatOllama(model="qwen3:4b", temperature=0)

def clean_llm_output(raw_text: str) -> str:
    """
    Removes unwanted <think> tags and their content from the LLM's output.
    """
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

prompt_template = """You are a professional news summarizer. Your task is to provide a concise and factual summary of the provided article. Do not include your own opinions or analysis.

{revision_instructions}

Return ONLY the summary itself.

**Full Article Text:**
```{original_text}```
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
summarizer_chain = prompt | llm | StrOutputParser()

def summarize_node(state: GraphState) -> dict:
    """
    Generates a summary or revision based on the current state.
    """
    print("---  Summarizing... ---")
    revision_number = state["revision_number"]
    original_text = state["original_text"]
    
    if revision_number == 0:
        revision_instructions = "Please write a concise, neutral summary of the article provided below."
    else:
        revision_instructions = f"""This is revision number {revision_number}.
Please revise the following summary based on the user's feedback.

**Previous Summary:**
{state['current_summary']}

**User Feedback:**
"{state['user_feedback']}"

Your task is to generate a new, improved summary.
"""

    new_summary = summarizer_chain.invoke({
        "original_text": original_text,
        "revision_instructions": revision_instructions,
    })
    clean_draft = clean_llm_output(new_summary)

    print("---  Summary Generated ---")
    
    return {
        "current_summary": clean_draft,
        "revision_number": revision_number + 1
    }

def human_review_node(state: GraphState) -> dict:
    """A placeholder node where the graph will stop to wait for human input."""
    print("--- Awaiting Human Review ---")
    return {}

def should_continue(state: GraphState) -> str:
    """Determines whether to continue revising or to end the process."""
    if state["user_feedback"].lower() == "approve":
        print("---  Summary Approved by User ---")
        return "end"
    else:
        print("--- Revising Summary Based on Feedback ---")
        return "continue"

workflow = StateGraph(GraphState)

workflow.add_node("summarize", summarize_node)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("summarize")
workflow.add_edge("summarize", "human_review")
workflow.add_conditional_edges(
    "human_review",
    should_continue,
    {"continue": "summarize", "end": END}
)

app = workflow.compile(interrupt_before=["human_review"])


if __name__ == "__main__":
    print("Welcome to the Interactive News Summarizer!")

    source_input= "news/news1.txt"
    
    if source_input.startswith("http"):
        text = get_text_from_url(source_input)
    else:
        text = get_text_from_file(source_input)

    if text.startswith("Error:"):
        print(text)
    else:
        initial_state = {
            "source": source_input,
            "original_text": text,
            "current_summary": "",
            "user_feedback": "",
            "revision_number": 0,
        }
        while True:
            result = app.invoke(initial_state)
            latest_summary = result["current_summary"]
            print("\n\n--- LATEST SUMMARY ---")
            print(latest_summary)
            print("----------------------")
            feedback = input(
                'Your feedback (e.g., "focus on the financial impact", "make it shorter", or type "approve" to finish): '
            )
            if feedback.lower() == "approve":
                print("\n Final summary approved!")
                print("\n=====Final Summary=====")
                print("\n",latest_summary)
                break
            initial_state = result
            initial_state["user_feedback"] = feedback
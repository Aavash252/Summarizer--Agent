import os
import re  
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# --- 1. Define the State for our Graph ---
class GraphState(TypedDict):
    topic: str
    audience: str
    draft_email: str
    human_feedback: str
    revision_number: int

# --- 2. Set up the LLM and Chains ---
llm = ChatOllama(model="qwen3:4b", temperature=0)

prompt_template = """You are an expert email writer. Your task is to craft a professional and clear email. Do not include any of your own reasoning or thoughts in the output, only the email itself.

**Email Topic:** {topic}
**Target Audience:** {audience}

{revision_instructions}

Please generate the complete email draft now.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
email_writer_chain = prompt | llm | StrOutputParser()

# --- 3. Add a Helper Function to Clean the Output ---
def clean_llm_output(raw_text: str) -> str:
    """
    Removes unwanted <think> tags and their content from the LLM's output.
    """
    # This regex finds any <think>...</think> block and removes it.
    # re.DOTALL allows '.' to match newline characters, handling multi-line thoughts.
    # re.IGNORECASE makes the pattern case-insensitive (e.g., matches <Think>).
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

# --- 4. Define the Nodes of the Graph ---
def draft_email_node(state: GraphState) -> dict:
    """
    Generates an email draft based on the current state and cleans the output.
    """
    print("--- Drafting Email ---")
    revision_number = state["revision_number"]

    if revision_number == 0:
        revision_instructions = "This is the first draft. Please write the email from scratch."
    else:
        revision_instructions = f"""This is revision number {revision_number}.
Please revise the following draft based on the user's feedback.
**Previous Draft:**
{state['draft_email']}

**User Feedback:**
{state['human_feedback']}
"""
    # Call the LLM chain to get the raw draft
    raw_draft = email_writer_chain.invoke({
        "topic": state["topic"],
        "audience": state["audience"],
        "revision_instructions": revision_instructions,
    })

    # **THE FIX IS HERE:** Clean the raw output before saving it to the state.
    clean_draft = clean_llm_output(raw_draft)

    print("--- Draft Complete ---")
    
    return {
        "draft_email": clean_draft,  # Use the cleaned draft
        "revision_number": revision_number + 1
    }

def request_human_review_node(state: GraphState) -> dict:
    print("--- ✋ Awaiting Human Review ---")
    return {}

# --- 5. Define the Conditional Edge Logic ---
def should_continue(state: GraphState) -> str:
    if state["human_feedback"].lower() == "approve":
        print("---  Email Approved by User ---")
        return "end"
    else:
        print("--- Revising Draft Based on Feedback ---")
        return "continue"

# --- 6. Build and Compile the Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("draft_email", draft_email_node)
workflow.add_node("human_review", request_human_review_node)
workflow.set_entry_point("draft_email")
workflow.add_edge("draft_email", "human_review")
workflow.add_conditional_edges(
    "human_review",
    should_continue,
    {"continue": "draft_email", "end": END}
)
app = workflow.compile(interrupt_before=["human_review"])

def save_email_to_file(topic: str, content: str):
    """
    Saves the final email content to a .txt file with a sanitized filename.
    """
    # Sanitize the topic to create a safe and valid filename
    # Replaces spaces with underscores and removes characters that aren't letters, numbers, or underscores
    sanitized_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
    filename = f"email_{sanitized_topic}.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n Final email successfully saved to: {filename}")
    except IOError as e:
        print(f"\n Error: Could not save the file. Reason: {e}")


# --- 7. The Main Application Loop ---
if __name__ == "__main__":
    print("🚀 Welcome to the Human-in-the-Loop Email Drafter!")

    topic = input("What is the topic of the email? ")
    audience = input("Who is the audience? ")
    
    current_state = {
        "topic": topic,
        "audience": audience,
        "draft_email": "",
        "human_feedback": "",
        "revision_number": 0,
    }
    
    while True:
        result = app.invoke(current_state)
        latest_draft = result["draft_email"]
        
        print("\n\n--- LATEST DRAFT ---")
        print(latest_draft)
        print("--------------------")

        feedback = input(
            "Your feedback (type 'approve' to finish, 'reject', or provide specific edits): "
        )

        if feedback.lower() == "approve":
            print("\n🎉 Final email approved!")
            save_email_to_file(topic, latest_draft)

            break
        
        current_state = result
        current_state["human_feedback"] = feedback
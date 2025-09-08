import os
import re
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Define the State for our Graph ---
class GraphState(TypedDict):
    topic: str
    audience: str
    draft_email: str
    human_feedback: str
    revision_number: int
    final_email: str

# --- 2. Set up the LLM and Chains for our Agents ---
llm = ChatOllama(model="qwen3:4b", temperature=0)
draft_prompt_template = """You are an expert email writer. Your task is to craft a professional and clear email. Do not include any of your own reasoning or thoughts like <think> tags in the output. Output ONLY the email text itself.
**Email Topic:** {topic}
**Target Audience:** {audience}
{revision_instructions}
Please generate the complete email draft now."""
draft_prompt = ChatPromptTemplate.from_template(draft_prompt_template)
email_writer_chain = draft_prompt | llm | StrOutputParser()
review_prompt_template = """You are a meticulous editor. Your task is to review the following email draft for any grammatical errors, spelling mistakes, awkward phrasing, or issues with professional tone.
Please make any necessary corrections to produce a final, polished version.
Do not add any commentary, notes, or introductions. Output ONLY the final, corrected email text.
**Email Draft to Review:**
---
{draft_email}
---
"""
review_prompt = ChatPromptTemplate.from_template(review_prompt_template)
email_reviewer_chain = review_prompt | llm | StrOutputParser()

# --- Helper Function to Clean the Output ---
def clean_llm_output(raw_text: str) -> str:
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

# --- 3. Define the Nodes of the Graph ---
def draft_email_node(state: GraphState) -> dict:
    print("--- 📝 Drafting Email ---")
    revision_number = state["revision_number"]
    if revision_number == 0:
        revision_instructions = "This is the first draft. Please write the email from scratch."
    else:
        revision_instructions = f"""This is revision number {revision_number}.
Please revise the following draft based on the user's feedback.
**Previous Draft:**\n{state['draft_email']}\n\n**User Feedback:**\n{state['human_feedback']}"""
    raw_draft = email_writer_chain.invoke({
        "topic": state["topic"],
        "audience": state["audience"],
        "revision_instructions": revision_instructions,
    })
    clean_draft = clean_llm_output(raw_draft)
    print("--- ✅ Draft Complete ---")
    return {"draft_email": clean_draft, "revision_number": state["revision_number"] + 1}

def request_human_review_node(state: GraphState) -> dict:
    print("--- ✋ Awaiting Human Review ---")
    return {}

def final_review_node(state: GraphState) -> dict:
    print("--- 🧐 Running Final Review and Polish ---")
    final_draft = state["draft_email"]
    polished_email = email_reviewer_chain.invoke({"draft_email": final_draft})
    clean_polished_email = clean_llm_output(polished_email)
    filename = "final_email.txt"
    with open(filename, "w") as f:
        f.write(clean_polished_email)
    print(f"--- 💾 Final Email Saved to {filename} ---")
    return {"final_email": clean_polished_email}

# --- 4. Define the Conditional Edge Logic ---
def should_continue(state: GraphState) -> str:
    if state["human_feedback"].strip().lower() == "approve":
        return "final_review"
    else:
        return "continue"

# --- 5. Build and Compile the Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("draft_email", draft_email_node)
workflow.add_node("human_review", request_human_review_node)
workflow.add_node("final_review", final_review_node)
workflow.set_entry_point("draft_email")
workflow.add_edge("draft_email", "human_review")
workflow.add_edge("final_review", END)
workflow.add_conditional_edges(
    "human_review",
    should_continue,
    {"continue": "draft_email", "final_review": "final_review"}
)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_after=["human_review"])


# --- 6. The Main Application Loop (Canonical Checkpointer Pattern) ---
if __name__ == "__main__":
    print("🚀 Welcome to the Human-in-the-Loop Email Drafter!")

    topic = input("What is the topic of the email? ")
    audience = input("Who is the audience? ")
    
    thread_config = {"configurable": {"thread_id": "email-thread-1"}}
    
    initial_input = {
        "topic": topic,
        "audience": audience,
        "revision_number": 0,
        "human_feedback": "",
    }
    
    # Kick off the graph. It will run until the first interruption.
    app.invoke(initial_input, thread_config)

    while True:
        # Get the state of the graph at the pause point.
        snapshot = app.get_state(thread_config)
        
        # Check if the graph has finished.
        if not snapshot.next: # The 'next' field is empty when the graph is at the END
            print("\n🎉 Process complete! Final email has been reviewed and saved.")
            break

        latest_draft = snapshot.values["draft_email"]
        # print("\n\n--- LATEST DRAFT ---")
        print(latest_draft)
        # print("--------------------")
        
        feedback = input(
            "Your feedback (type 'approve' to finish, 'reject', or provide specific edits): "
        )
        
        # **THE FIX IS HERE:**
        # Instead of calling invoke with new data, we explicitly update the state
        # in the checkpointer, and then call invoke with no data to resume.
        app.update_state(
            thread_config,
            {"human_feedback": feedback},
        )
        # Resume the graph from the updated state.
        app.invoke(None, thread_config)
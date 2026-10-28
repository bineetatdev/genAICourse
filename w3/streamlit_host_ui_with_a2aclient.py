
import streamlit as st
import asyncio
from datetime import datetime
import sys
import os

# Adjust the path to include your src folder
sys.path.append("/Users/bineet/Library/CloudStorage/OneDrive-HitachiGroup/Apps/ai-agents/smart-bems/src")

from client.client import A2AClient
from models.agent import AgentCard
from models.task import Task

# --- Agent metadata (for icons and names) ---
AGENT_ICONS = {
    "DemandAgent": "💡",
    "GreetingAgent": "👋",
    "OrchestratorAgent": "🧠"
}

# --- Orchestrator Config ---
ORCHESTRATOR_URL = "http://localhost:10002"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Multi-Agent Chat", layout="centered")
st.title("🧠 Orchestrated Agent Chat")
st.markdown("Chat with a multi-agent system powered by Gemini. All agent interactions are tracked below.")

# Session state for message history
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Chat UI display
for log in st.session_state.chat_log:
    with st.chat_message(log["sender"]):
        icon = AGENT_ICONS.get(log["agent"], "🤖")
        st.markdown(f"{icon} **{log['agent']}**: {log['text']}")

# User input box
user_input = st.chat_input("Ask your question...")

# --- Async task sender using A2AClient ---
async def send_task_to_orchestrator(message, session_id="web-ui-session") -> Task:
    client = A2AClient(url=ORCHESTRATOR_URL)
    payload = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "sessionId": session_id,
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": message}
            ]
        }
    }
    task = await client.send_task(payload)
    return task

# --- Handle Input ---
if user_input:
    st.session_state.chat_log.append({
        "sender": "user",
        "agent": "You",
        "text": user_input
    })

    async def handle_task():
        with st.spinner("OrchestratorAgent is processing..."):
            task: Task = await send_task_to_orchestrator(user_input)
            for msg in task.history:
                role = msg.role
                parts = msg.parts
                text = parts[0].text if parts else ""
                agent_name = "Unknown"

                if role == "agent" and "DemandAgent" in text:
                    agent_name = "DemandAgent"
                elif role == "agent" and "Hello" in text:
                    agent_name = "GreetingAgent"
                elif role == "agent":
                    agent_name = "OrchestratorAgent"

                st.session_state.chat_log.append({
                    "sender": role,
                    "agent": agent_name,
                    "text": text
                })

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(handle_task())

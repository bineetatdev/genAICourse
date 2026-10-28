
import streamlit as st
import asyncio
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO

# --- Agent metadata (for icons and names) ---
AGENT_ICONS = {
    "DemandAgent": "💡",
    "GreetingAgent": "👋",
    "OrchestratorAgent": "🧠"
}

# --- Orchestrator Config ---
ORCHESTRATOR_URL = "http://localhost:10002/tasks/send"

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

# --- Helper: Send JSON-RPC task to Orchestrator ---
async def send_to_orchestrator(message, session_id="web-ui-session"):
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
    try:
        with st.spinner("Routing through OrchestratorAgent..."):
            response = requests.post(ORCHESTRATOR_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            task = data.get("result", {})
            messages = task.get("history", [])

            # Add each agent message to chat log
            for msg in messages:
                role = msg.get("role", "agent")
                parts = msg.get("parts", [])
                text = parts[0].get("text", "") if parts else ""
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
    except Exception as e:
        st.error(f"Error contacting orchestrator: {e}")

# --- Handle Input ---
if user_input:
    # Log user input
    st.session_state.chat_log.append({
        "sender": "user",
        "agent": "You",
        "text": user_input
    })

    # Send request asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_to_orchestrator(user_input))

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")


# Define a simple tool
@tool
def get_answer(query: str) -> str:
    """A simple tool that returns a fixed answer."""
    return f"The answer to '{query}' is 42."


tools = [get_answer]


def get_smart_model(message_count: int):
    """Select appropriate model based on conversation state."""
    print(f"--- Message count: {message_count} ---")
    # For short conversations, use a smaller, faster model
    if message_count <= 2:
        print("Using model: llama-3.1-8b-instant")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

    # For longer conversations, use a more capable model
    print("Using model: llama-3.3-70b-versatile")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


# Track conversation
conversation_history = []
message_count = 0


def chat(user_input: str):
    global message_count, conversation_history

    message_count += 1
    conversation_history.append({"role": "user", "content": user_input})

    # Select model dynamically
    current_model = get_smart_model(message_count)

    # Create agent with selected model
    agent = create_agent(
        model=current_model, tools=tools, system_prompt="You are a helpful assistant."
    )

    result = agent.invoke({"messages": conversation_history})

    response_content = result['messages'][-1].content
    conversation_history.append({"role": "assistant", "content": response_content})
    return response_content


# Usage
print(f"AI: {chat('Hello!')}")
print(f"\nAI: {chat('Tell me more')}")
print(f"\nAI: {chat('Give me a complex answer about the universe.')}")

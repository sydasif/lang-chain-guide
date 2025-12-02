import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq language model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


# Define some tools for the agent to use
@tool
def get_user_info(user_id: str) -> str:
    """Pretends to look up user info."""
    if user_id == "user-session-abc123":
        return "User info: Name is Alice, lives in Seattle."
    return "User not found."


tools = [get_user_info]

# Create agent with the new API - memory/checkpointing would typically be handled
# via state management in the new API, but for this example we'll manage conversation history manually

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

# Since the new API handles memory differently, we'll simulate the conversation
# by including all messages in the call

# First message
print("User: My name is Alice and I live in Seattle")
messages = [
    {"role": "user", "content": "My name is Alice and I live in Seattle"}
]
result1 = agent.invoke({"messages": messages})
print(f"AI: {result1['messages'][-1].content}")

# Add the AI response to the message history
messages.append(result1['messages'][-1])  # AI response
messages.append({"role": "user", "content": "What's my name and where do I live?"})  # New user query

# Second message - the agent now has context from the conversation history
print("\nUser: What's my name and where do I live?")
result2 = agent.invoke({"messages": messages})
print(f"AI: {result2['messages'][-1].content}")

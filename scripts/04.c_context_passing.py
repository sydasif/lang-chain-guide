import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()


# Initialize the ChatGroq language model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


# Tool that uses context
@tool
def locate_user(user_id: str) -> str:
    """Find a user's location based on their ID."""
    # In production, this would query a database
    users_db = {
        "user_123": "New York",
        "user_456": "Los Angeles",
        "user_789": "Chicago",
    }
    return users_db.get(user_id, "Unknown location")


@tool
def get_local_weather(location: str) -> str:
    """Get weather for a specific location."""
    return f"Weather in {location}: Partly cloudy, 68°F"


# Create agent with multiple tools
tools = [locate_user, get_local_weather]
agent = create_agent(
    model=llm, tools=tools, system_prompt="You are a helpful assistant."
)

# Invoke the agent - in the new API, context passing might be different
# For this simplified version, we'll demonstrate a direct call
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Locate user user_123 and then get the weather for their location",
        }
    ]
})

print(result["messages"][-1].content)

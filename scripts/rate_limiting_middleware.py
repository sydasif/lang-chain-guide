import os
import time
from datetime import datetime

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


# Define a simple tool
@tool
def simple_tool():
    """A simple tool."""
    return "This is a simple tool."


tools = [simple_tool]

# Rate limiting functionality needs to be implemented at the application level
# since the callback system is different in the new API
class RateLimiter:
    def __init__(self, max_calls_per_minute=5):
        self.max_calls = max_calls_per_minute
        self.calls = []

    def check_rate_limit(self):
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if (now - t).seconds < 60]

        if len(self.calls) >= self.max_calls:
            raise Exception("Rate limit exceeded. Please wait.")

        print(f"Call #{len(self.calls) + 1} accepted.")
        self.calls.append(now)


# Usage
rate_limiter = RateLimiter(max_calls_per_minute=5)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

# Make calls that should succeed - apply rate limiting at the application level
for i in range(5):
    try:
        rate_limiter.check_rate_limit()
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"This is call {i + 1}"}]
        })
        print(f"Response: {result['messages'][-1].content}")
        time.sleep(1)  # Stagger calls slightly
    except Exception as e:
        print(e)

# This call should fail
print("\n--- Making a call that should exceed the rate limit ---")
try:
    rate_limiter.check_rate_limit()
    result = agent.invoke({
        "messages": [{"role": "user", "content": "This call should fail"}]
    })
    print(f"Response: {result['messages'][-1].content}")
except Exception as e:
    print(f"Caught expected exception: {e}")

# Wait for the rate limit to reset
print("\n--- Waiting for 60 seconds for rate limit to reset ---")
time.sleep(60)

# This call should succeed again
print("\n--- Making a call after waiting ---")
try:
    rate_limiter.check_rate_limit()
    result = agent.invoke({
        "messages": [{"role": "user", "content": "This call should succeed"}]
    })
    print(f"Response: {result['messages'][-1].content}")
except Exception as e:
    print(e)

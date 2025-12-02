import os
from datetime import datetime

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
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "seattle" in city.lower():
        return "sunny and 75°F"
    if "boston" in city.lower():
        return "cloudy and 68°F"
    return "weather is unknown for that location."


tools = [get_weather]

# Initialize the ChatGroq language model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Create agent with the new API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

# Make some calls - since callbacks are different in the new API,
# we'll implement simple logging around the calls
call_log = []

print("\n--- First call ---")
start_time = datetime.now()
print(f"\n🤖 AI Started at {start_time.strftime('%H:%M:%S')}")
print("📝 Processing prompt: What's the weather in Seattle?")

result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seattle?"}]
})

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(f"✅ AI Completed in {duration:.2f} seconds")
print(f"🔧 Tool calls and processing completed")
call_log.append({
    "timestamp": end_time.isoformat(),
    "duration": duration,
    "prompt": "What's the weather in Seattle?"
})

print(result1["messages"][-1].content)


print("\n--- Second call ---")
start_time = datetime.now()
print(f"\n🤖 AI Started at {start_time.strftime('%H:%M:%S')}")
print("📝 Processing prompt: And in Boston?")

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "And in Boston?"}]
})

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(f"✅ AI Completed in {duration:.2f} seconds")
print(f"🔧 Tool calls and processing completed")
call_log.append({
    "timestamp": end_time.isoformat(),
    "duration": duration,
    "prompt": "And in Boston?"
})

print(result2["messages"][-1].content)

# View analytics
if call_log:
    total_calls = len(call_log)
    avg_duration = sum(call["duration"] for call in call_log) / total_calls

    analytics = f"""
Analytics Summary:
- Total Calls: {total_calls}
- Average Duration: {avg_duration:.2f} seconds
- Total Time: {sum(call["duration"] for call in call_log):.2f} seconds
    """
    print(analytics)
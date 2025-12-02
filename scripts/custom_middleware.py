"""
Custom Middleware Example - LangChain Zero-to-Hero Part 5

This script demonstrates how to create a custom middleware class using
AgentMiddleware to log agent activities and performance metrics.
"""

from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Custom Middleware Demo")
print("=" * 60)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "seattle" in city.lower():
        return "sunny and 75°F"
    if "boston" in city.lower():
        return "cloudy and 68°F"
    return "weather is unknown for that location."


# Custom middleware for performance tracking
class PerformanceMiddleware(AgentMiddleware):
    """Middleware to track performance metrics."""

    def __init__(self):
        self.call_log = []
        self.start_time = None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Record start time."""
        self.start_time = datetime.now()
        print(f"\n🤖 AI Started at {self.start_time.strftime('%H:%M:%S')}")
        print(f"📝 Processing prompt: {state['messages'][-1].content}")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Record duration and log metrics."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        print(f"✅ AI Completed in {duration:.2f} seconds")
        print("🔧 Tool calls and processing completed")

        self.call_log.append({
            "timestamp": end_time.isoformat(),
            "duration": duration,
            "prompt": state['messages'][-1].content,
        })
        return None

    def get_analytics(self):
        """Calculate and print analytics."""
        if not self.call_log:
            return "No calls recorded."

        total_calls = len(self.call_log)
        avg_duration = sum(call["duration"] for call in self.call_log) / total_calls

        return f"""
Analytics Summary:
- Total Calls: {total_calls}
- Average Duration: {avg_duration:.2f} seconds
- Total Time: {sum(call["duration"] for call in self.call_log):.2f} seconds
        """


# Initialize middleware
perf_middleware = PerformanceMiddleware()

# Create agent
print("\n[Step 1] Creating agent with performance middleware...")
agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[perf_middleware],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created")

# Test 1
print("\n--- First call ---")
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seattle?"}]
})
print(result1["messages"][-1].content)

# Test 2
print("\n--- Second call ---")
result2 = agent.invoke({"messages": [{"role": "user", "content": "And in Boston?"}]})
print(result2["messages"][-1].content)

# View analytics
print("\n" + "=" * 60)
print(perf_middleware.get_analytics())
print("=" * 60)

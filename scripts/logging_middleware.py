"""
Logging Middleware Example - LangChain Zero-to-Hero Part 5

This script demonstrates the official AgentMiddleware class for logging
agent behavior with before_model and after_model hooks.
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
print("Logging Middleware Demo - Official AgentMiddleware")
print("=" * 60)


# Define a simple tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "seattle" in city.lower():
        return "sunny and 75°F"
    if "boston" in city.lower():
        return "cloudy and 68°F"
    return "weather is unknown for that location."


# Create official logging middleware using AgentMiddleware class
class LoggingMiddleware(AgentMiddleware):
    """Official middleware for logging agent behavior."""

    def __init__(self):
        self.call_count = 0
        self.start_times = {}

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Called once when agent starts."""
        print("\n🚀 Agent session started")
        print(f"📝 Initial query: {state['messages'][0].content}")
        return None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Called before each model invocation."""
        self.call_count += 1
        self.start_times[self.call_count] = datetime.now()

        print(f"\n🤖 Model Call #{self.call_count}")
        print(f"📊 Messages in context: {len(state['messages'])}")
        print(
            f"⏰ Started at: {self.start_times[self.call_count].strftime('%H:%M:%S')}"
        )
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called after each model response."""
        end_time = datetime.now()
        duration = (end_time - self.start_times[self.call_count]).total_seconds()

        last_message = state['messages'][-1]
        content_preview = (
            last_message.content[:80] if last_message.content else "No content"
        )

        print(f"✅ Model Call #{self.call_count} completed")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"💬 Response preview: {content_preview}...")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called once when agent completes."""
        print("\n🏁 Agent session completed")
        print(f"📈 Total model calls: {self.call_count}")
        return None


# Create agent with logging middleware
print("\n[Step 1] Creating agent with LoggingMiddleware...")
agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[LoggingMiddleware()],
    system_prompt="You are a helpful weather assistant.",
)
print("✓ Agent created with middleware")

# Test 1: Simple query
print("\n" + "=" * 60)
print("Test 1: Simple Weather Query")
print("=" * 60)

result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seattle?"}]
})

print(f"\n📤 Final Response:\n{result1['messages'][-1].content}")

# Test 2: Another query
print("\n" + "=" * 60)
print("Test 2: Another Location")
print("=" * 60)

result2 = agent.invoke({"messages": [{"role": "user", "content": "And in Boston?"}]})

print(f"\n📤 Final Response:\n{result2['messages'][-1].content}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- AgentMiddleware provides official hooks for logging")
print("- before_agent/after_agent run once per session")
print("- before_model/after_model run for each model call")
print("- Middleware can maintain state across calls")
print("- Perfect for observability in production")

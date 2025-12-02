"""
Decorator Middleware Example - LangChain Zero-to-Hero Part 5

This script demonstrates decorator-based middleware for simple use cases
using @before_model and @after_model decorators.
"""

from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    after_agent,
    after_model,
    before_agent,
    before_model,
)
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Decorator Middleware Demo")
print("=" * 60)


# Define a simple tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Define middleware using decorators
@before_agent
def session_start(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log when agent session starts."""
    print("\n🚀 Session started")
    return None


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log before each model call."""
    print(f"\n📝 Calling model with {len(state['messages'])} messages")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log after each model call."""
    last_msg = state['messages'][-1]
    preview = last_msg.content[:60] if last_msg.content else "No content"
    print(f"✅ Model responded: {preview}...")
    return None


@after_agent
def session_end(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log when agent session ends."""
    print("\n🏁 Session completed")
    return None


# Create agent with decorator-based middleware
print("\n[Step 1] Creating agent with decorator middleware...")
agent = create_agent(
    model=llm,
    tools=[calculate],
    middleware=[session_start, log_before_model, log_after_model, session_end],
    system_prompt="You are a helpful math assistant.",
)
print("✓ Agent created")

# Test 1: Simple calculation
print("\n" + "=" * 60)
print("Test 1: Simple Math")
print("=" * 60)

result1 = agent.invoke({"messages": [{"role": "user", "content": "What is 15 * 23?"}]})

print(f"\n📤 Final Answer: {result1['messages'][-1].content}")

# Test 2: Complex calculation
print("\n" + "=" * 60)
print("Test 2: Complex Math")
print("=" * 60)

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate (100 + 50) / 3"}]
})

print(f"\n📤 Final Answer: {result2['messages'][-1].content}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- Decorators (@before_model, @after_model) are quick and simple")
print("- Perfect for stateless logging")
print("- Can combine multiple decorators")
print("- Less code than class-based middleware")
print("- Great for prototyping")

"""
Rate Limiting Middleware - LangChain Zero-to-Hero Part 5

This script demonstrates rate limiting using the official AgentMiddleware class
to prevent excessive API calls and control costs.
"""

import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("Rate Limiting Middleware Demo")
print("=" * 60)


@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    return f"Found results for: {query}"


# Official rate limiting middleware using AgentMiddleware
class RateLimitMiddleware(AgentMiddleware):
    """Middleware to limit the number of model calls per minute."""

    def __init__(self, max_calls_per_minute: int = 5):
        self.max_calls = max_calls_per_minute
        self.calls = []
        print(f"⚙️  Rate limit set to {max_calls_per_minute} calls/minute")

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check rate limit before each model call."""
        now = datetime.now()

        # Remove calls older than 60 seconds
        self.calls = [t for t in self.calls if (now - t).total_seconds() < 60]

        # Check if limit exceeded
        if len(self.calls) >= self.max_calls:
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call).total_seconds()

            print("\n🚫 Rate limit exceeded!")
            print(f"⏳ Please wait {wait_time:.1f} seconds")
            raise Exception(
                f"Rate limit exceeded. "
                f"Maximum {self.max_calls} calls per minute. "
                f"Please wait {wait_time:.1f} seconds."
            )

        # Record this call
        self.calls.append(now)
        remaining = self.max_calls - len(self.calls)
        print(f"\n✅ Rate limit check passed ({remaining} calls remaining)")

        return None


# Create agent with rate limiting
print("\n[Step 1] Creating agent with rate limiting...")
agent = create_agent(
    model=llm,
    tools=[search_database],
    middleware=[RateLimitMiddleware(max_calls_per_minute=3)],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created with rate limit: 3 calls/minute")

# Test: Make multiple calls to trigger rate limit
print("\n" + "=" * 60)
print("Testing Rate Limit (3 calls/minute)")
print("=" * 60)

for i in range(1, 6):
    print(f"\n--- Call {i} ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Search for item {i}"}]
        })
        print(f"✅ Call {i} succeeded")
        print(f"Response: {result['messages'][-1].content[:60]}...")
    except Exception as e:
        print(f"❌ Call {i} failed: {str(e)}")
        # Don't break, continue to show it keeps failing until reset
        # But for demo speed, we'll stop after first failure
        break

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- Rate limiting prevents excessive API calls")
print("- Protects against cost spikes")
print("- Configurable limits (calls per minute)")
print("- Automatic tracking with before_model hook")
print("- Production-ready pattern")

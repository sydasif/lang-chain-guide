"""
Summarization Middleware - LangChain Zero-to-Hero Part 5

This script demonstrates how to use AgentMiddleware to automatically summarize
conversation history when it gets too long, keeping the context window manageable.
"""

from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("Summarization Middleware Demo")
print("=" * 60)


@tool
def get_info(topic: str) -> str:
    """Gets information on a topic."""
    return f"Information about {topic}."


# Official summarization middleware
class SummarizationMiddleware(AgentMiddleware):
    """Middleware to summarize history when it exceeds a threshold."""

    def __init__(self, model, max_messages: int = 5):
        self.model = model
        self.max_messages = max_messages

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check history length and summarize if needed."""
        messages = state["messages"]

        if len(messages) > self.max_messages:
            print(f"\n🗜️  History too long ({len(messages)} msgs). Summarizing...")

            # Create summarization prompt
            summary_prompt = (
                "Summarize the following conversation history concisely, "
                "preserving key facts and context:\n\n" + str(messages)
            )

            # Generate summary
            summary = self.model.invoke(summary_prompt).content
            print(f"📝 Summary generated: {summary[:50]}...")

            # Replace history with summary
            # Note: In a real app, you might keep the last few messages
            new_messages = [
                SystemMessage(content=f"Previous conversation summary: {summary}"),
                messages[-1],  # Keep the last user message
            ]

            # Update state
            state["messages"] = new_messages
            print(f"✅ History compressed to {len(new_messages)} messages")

        return None


# Create agent with summarization
print("\n[Step 1] Creating agent with summarization (max 4 messages)...")
agent = create_agent(
    model=llm,
    tools=[get_info],
    middleware=[SummarizationMiddleware(model=llm, max_messages=4)],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created")

# Test: Run a long conversation
print("\n" + "=" * 60)
print("Testing Automatic Summarization")
print("=" * 60)

messages = []

# Simulate a conversation
for i in range(1, 7):
    print(f"\n--- Turn {i} ---")
    user_msg = {"role": "user", "content": f"Tell me about topic {i}"}

    # Add user message to history
    messages.append(user_msg)

    result = agent.invoke({"messages": messages})

    # Update our local history with the state from the agent
    # This captures the summarization if it happened
    messages = result["messages"]

    response = messages[-1]
    print(f"AI: {response.content}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- Middleware monitors message history length")
print("- Automatically triggers summarization when threshold reached")
print("- Modifies state (messages) before model sees it")
print("- Keeps context window small while preserving memory")

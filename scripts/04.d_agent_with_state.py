"""
AgentState Example - LangChain Zero-to-Hero Part 3

This script demonstrates using AgentState to track custom state beyond
just conversation messages, following official LangChain patterns.
"""

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("AgentState Demo - Custom State Management")
print("=" * 60)


# Define custom state that extends AgentState
class UserContextState(AgentState):
    """Custom state that includes user context beyond messages."""

    user_name: str
    user_role: str
    preferences: dict


# Create agent with custom state schema (no tools for simplicity)
print("\n[Step 1] Creating agent with custom state schema...")
agent = create_agent(
    model=llm,
    tools=[],
    state_schema=UserContextState,
    system_prompt=(
        "You are a helpful assistant. "
        "Greet the user by name and acknowledge their role. "
        "Personalize your responses based on their context."
    ),
)
print("✓ Agent created with UserContextState")

# Test 1: Engineer user
print("\n" + "=" * 60)
print("Test 1: Engineer User")
print("=" * 60)

result1 = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}],
    "user_name": "Alice",
    "user_role": "engineer",
    "preferences": {"theme": "dark", "notifications": True},
})

print("\nUser: Alice (engineer)")
print(f"AI: {result1['messages'][-1].content}")

# Test 2: Admin user
print("\n" + "=" * 60)
print("Test 2: Admin User")
print("=" * 60)

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "What can I help you with today?"}],
    "user_name": "Bob",
    "user_role": "admin",
    "preferences": {"theme": "light", "notifications": False},
})

print("\nUser: Bob (admin)")
print(f"AI: {result2['messages'][-1].content}")

# Test 3: Analyst user with conversation history
print("\n" + "=" * 60)
print("Test 3: Analyst User with Conversation History")
print("=" * 60)

# Simulate a conversation with history
messages = [{"role": "user", "content": "Hi there!"}]

result3 = agent.invoke({
    "messages": messages,
    "user_name": "Carol",
    "user_role": "analyst",
    "preferences": {"theme": "dark", "verbosity": "detailed"},
})

print("\nUser: Carol (analyst)")
print(f"Turn 1 - AI: {result3['messages'][-1].content}")

# Continue the conversation
messages.append(result3['messages'][-1])
messages.append({"role": "user", "content": "What's my role again?"})

result4 = agent.invoke({
    "messages": messages,
    "user_name": "Carol",
    "user_role": "analyst",
    "preferences": {"theme": "dark", "verbosity": "detailed"},
})

print(f"Turn 2 - AI: {result4['messages'][-1].content}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- AgentState extends beyond just messages")
print("- Custom state (user_name, user_role, preferences) is accessible")
print("- State persists across agent invocations")
print("- Type-safe and validated by Pydantic")
print("- Perfect for production apps with user context")

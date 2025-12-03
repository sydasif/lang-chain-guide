"""
Human-in-the-Loop Middleware - LangChain Zero-to-Hero Part 5

This script demonstrates how to use official AgentMiddleware to intercept
sensitive tool calls and require human approval before execution.
"""

from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("Human-in-the-Loop Middleware Demo")
print("=" * 60)


# Define tools
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email. This is a sensitive operation."""
    return f"Email sent to {recipient} with subject '{subject}'"


@tool
def check_calendar(date: str) -> str:
    """Checks the calendar for a given date."""
    return f"No events scheduled for {date}"


# Official middleware for human approval
class HumanApprovalMiddleware(AgentMiddleware):
    """Middleware to require human approval for sensitive tools."""

    def __init__(self, sensitive_tools: list[str]):
        self.sensitive_tools = sensitive_tools

    def wrap_tool_call(self, tool_call: Any, handler: Callable[[Any], Any]) -> Any:
        """Intercept tool calls and check for sensitive operations."""
        # Debugging: Print tool_call details
        print(f"\nDEBUG: tool_call type: {type(tool_call)}")
        print(f"DEBUG: tool_call dir: {dir(tool_call)}")
        print(f"DEBUG: tool_call: {tool_call}")

        # Try to access attributes safely based on inspection
        try:
            tool_name = getattr(tool_call, 'name', None) or tool_call.get('name')
            tool_args = getattr(tool_call, 'args', None) or tool_call.get('args')
        except Exception:
            tool_name = "unknown"
            tool_args = {}

        # Check if tool is sensitive
        if tool_name in self.sensitive_tools:
            print(f"\n⚠️  SENSITIVE ACTION DETECTED: {tool_name}")
            print(f"   Args: {tool_args}")

            # Request approval
            approval = input("   Do you approve this action? (yes/no): ")

            if approval.lower() != "yes":
                print("   ❌ Action rejected by user")
                return f"Error: User rejected the action '{tool_name}'."

            print("   ✅ Action approved")

        # Proceed with execution
        return handler(tool_call)


# Create agent with approval middleware
print("\n[Step 1] Creating agent with approval middleware...")
agent = create_agent(
    model=llm,
    tools=[send_email, check_calendar],
    middleware=[HumanApprovalMiddleware(sensitive_tools=["send_email"])],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created (send_email requires approval)")

# Test 1: Sensitive tool (should ask for approval)
print("\n" + "=" * 60)
print("Test 1: Sensitive Tool (send_email)")
print("=" * 60)
print("Note: Type 'yes' to approve, 'no' to reject")

try:
    result1 = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Send an email to boss@example.com saying 'I quit'",
            }
        ]
    })
    print(f"\nAI: {result1['messages'][-1].content}")
except Exception as e:
    print(f"\nError: {e}")

# Test 2: Safe tool (should run automatically)
print("\n" + "=" * 60)
print("Test 2: Safe Tool (check_calendar)")
print("=" * 60)

try:
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "Check my calendar for tomorrow"}]
    })
    print(f"\nAI: {result2['messages'][-1].content}")
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- wrap_tool_call intercepts execution before it happens")
print("- You can inspect tool name and arguments")
print("- Human can approve or reject specific actions")
print("- Safe tools run without interruption")

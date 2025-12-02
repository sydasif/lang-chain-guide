import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")


# Define some tools, one of which is sensitive
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email. This is a sensitive operation."""
    # In the new API, we'll implement human approval by asking for confirmation here
    print(f"Manual approval required for sensitive operation: send_email")
    print(f"  Recipient: {recipient}")
    print(f"  Subject: {subject}")
    print(f"  Body: {body}")
    approval = input("Approve this email? (yes/no): ")
    if approval.lower() != "yes":
        raise Exception(f"Operation 'send_email' cancelled by user")

    result = f"Email sent to {recipient} with subject '{subject}'"
    print(f"Executing tool 'send_email'...")
    return result


@tool
def check_calendar(date: str) -> str:
    """Checks the calendar for a given date."""
    print(f"Executing tool 'check_calendar'...")
    return f"No events scheduled for {date}"


tools = [send_email, check_calendar]

# Initialize the ChatGroq language model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Usage
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

# This call should trigger the approval prompt
print("\n--- Attempting to use a sensitive tool (send_email) ---")
try:
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Send an email to test@example.com with subject 'Hello' and body 'This is a test'",
            }
        ]
    })
    print(f"Result: {result['messages'][-1].content}")
except Exception as e:
    print(f"Caught exception: {e}")

# This call should NOT trigger the approval prompt
print("\n--- Attempting to use a non-sensitive tool (check_calendar) ---")
try:
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Check my calendar for tomorrow"}]
    })
    print(f"Result: {result['messages'][-1].content}")
except Exception as e:
    print(f"Caught exception: {e}")

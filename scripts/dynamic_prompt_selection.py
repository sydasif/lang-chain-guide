import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")


# Define a simple tool for the agent
@tool
def get_system_status() -> str:
    """Checks the system status."""
    return "System is running normally."


tools = [get_system_status]


def create_dynamic_agent(user_role: str):
    """Create an agent with role-specific instructions."""

    # Define different prompts for different roles
    prompts = {
        "admin": """You are an administrative assistant with elevated privileges.
                   You can access sensitive information and perform system operations.
                   Be professional and security-conscious.""",
        "customer": """You are a friendly customer service representative.
                      Help users with their questions and guide them to resources.
                      Always be polite and patient.""",
        "developer": """You are a technical assistant for developers.
                       Provide detailed code examples and technical explanations.
                       Use programming terminology appropriately.""",
    }

    system_prompt = prompts.get(user_role, prompts["customer"])

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


# Usage
print("--- Creating Admin Agent ---")
admin_agent = create_dynamic_agent("admin")
print("--- Creating Customer Agent ---")
customer_agent = create_dynamic_agent("customer")

# Same question, different responses based on role
print("\n--- Admin Agent Query ---")
print("User: How do I reset the system?")
admin_result = admin_agent.invoke({
    "messages": [{"role": "user", "content": "How do I reset the system?"}]
})
print(f"Admin AI: {admin_result['messages'][-1].content}")


print("\n--- Customer Agent Query ---")
print("User: How do I reset the system?")
customer_result = customer_agent.invoke({
    "messages": [{"role": "user", "content": "How do I reset the system?"}]
})
print(f"Customer AI: {customer_result['messages'][-1].content}")

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Retrieve the GROQ API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


def create_dynamic_agent(user_role: str):
    """
    Create an agent with role-specific instructions.

    Args:
        user_role (str): The role of the user (admin, customer, developer).

    Returns:
        Agent: A LangChain agent configured with the appropriate system prompt.
    """

    # Define different prompts for different roles
    # This is the core of the "Dynamic Prompt Selection" pattern
    prompts = {
        "admin": """You are an administrative assistant with elevated privileges.
                   You can access sensitive information and perform system operations.
                   Be professional, concise, and security-conscious.
                   Focus on system status, logs, and user management.""",
        "customer": """You are a friendly customer service representative.
                      Help users with their questions and guide them to resources.
                      Always be polite, patient, and empathetic.
                      Use simple language and avoid technical jargon.""",
        "developer": """You are a technical assistant for developers.
                       Provide detailed code examples, stack traces, and technical explanations.
                       Use programming terminology appropriately.
                       Assume the user is technically proficient.""",
    }

    # Select the appropriate prompt, defaulting to 'customer' if role is unknown
    system_prompt = prompts.get(user_role, prompts["customer"])

    print(f"\n[SYSTEM] Creating agent for role: '{user_role}'")
    print(f"[SYSTEM] Selected prompt: {system_prompt.splitlines()[0]}...")

    # Create the agent with the selected prompt
    # Note: We're passing an empty tool list [] for this specific demo to focus on the prompt
    agent = create_agent(model=llm, tools=[], system_prompt=system_prompt)
    return agent


# --- Demonstration ---

print("=" * 60)
print("DYNAMIC PROMPT SELECTION DEMONSTRATION")
print("=" * 60)

# The same query will be sent to agents with different personas
test_query = "I need to reset my password. How do I do that?"

roles_to_test = ["customer", "admin", "developer"]

for role in roles_to_test:
    print(f"\n--- Testing with Role: {role.upper()} ---")

    # 1. Create the agent dynamically based on the role
    agent = create_dynamic_agent(role)

    # 2. Send the query
    print(f"User: {test_query}")
    response = agent.invoke({"messages": [{"role": "user", "content": test_query}]})

    # 3. Print the response
    print(f"AI: {response['messages'][-1].content}")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("1. The SAME query produces DIFFERENT responses based on the system prompt")
print("2. 'Customer' agent is polite and helpful")
print("3. 'Admin' agent is professional and security-focused")
print("4. 'Developer' agent provides technical details (if applicable)")

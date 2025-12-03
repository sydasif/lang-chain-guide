import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# --- Step 1: Define Specialized Agents ---

# Technical Support Agent
tech_agent = create_agent(
    model=llm,
    tools=[],  # In a real app, this would have tech-specific tools like 'check_logs'
    system_prompt="""You are a specialized Technical Support Agent.
    Your expertise is in debugging, software installation, and error resolution.
    Provide detailed, technical steps to solve problems.
    If you don't know the answer, advise the user to check the documentation.""",
)

# Billing Support Agent
billing_agent = create_agent(
    model=llm,
    tools=[],  # In a real app, this would have billing tools like 'check_invoice'
    system_prompt="""You are a specialized Billing Support Agent.
    Your expertise is in invoices, payments, refunds, and subscription plans.
    Be polite, professional, and empathetic regarding financial matters.
    Verify transaction details before making statements.""",
)

# General Customer Service Agent
general_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""You are a General Customer Service Agent.
    Handle general inquiries, greetings, and non-technical, non-billing questions.
    If a query becomes too technical or billing-related, suggest transferring to a specialist.""",
)


# --- Step 2: Create the Router ---


def classify_query(query: str) -> str:
    """
    Classify the user's query into one of three categories: 'technical', 'billing', or 'general'.
    """
    query_lower = query.lower()

    # Keywords for classification (in production, you might use an LLM for this too)
    tech_keywords = [
        "error",
        "bug",
        "crash",
        "install",
        "setup",
        "code",
        "python",
        "api",
        "fail",
    ]
    billing_keywords = [
        "price",
        "cost",
        "invoice",
        "bill",
        "payment",
        "refund",
        "subscription",
        "credit card",
    ]

    if any(keyword in query_lower for keyword in tech_keywords):
        return "technical"
    if any(keyword in query_lower for keyword in billing_keywords):
        return "billing"
    return "general"


def route_and_respond(query: str):
    """
    Routes the query to the appropriate agent and returns the response.
    """
    # 1. Classify the query
    category = classify_query(query)
    print(f"\n[ROUTER] Query: '{query}'")
    print(f"[ROUTER] Classified as: {category.upper()}")

    # 2. Route to the specialized agent
    if category == "technical":
        print("[ROUTER] Handoff -> Technical Support Agent")
        response = tech_agent.invoke({"messages": [{"role": "user", "content": query}]})
    elif category == "billing":
        print("[ROUTER] Handoff -> Billing Support Agent")
        response = billing_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
    else:
        print("[ROUTER] Handoff -> General Agent")
        response = general_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

    return response["messages"][-1].content


# --- Step 3: Demonstration ---

print("=" * 60)
print("AGENT ROUTING DEMONSTRATION")
print("=" * 60)

test_queries = [
    "My installation failed with error code 500",
    "I need a refund for my last month's subscription",
    "What are your business hours?",
    "How do I update the API key in the configuration?",
]

for query in test_queries:
    response = route_and_respond(query)
    print(f"Agent Response: {response}\n" + "-" * 60)

print("\nKEY TAKEAWAYS:")
print("1. Specialized agents perform better than one generic agent")
print("2. Routing logic (classifier) sits in front of the agents")
print("3. Each agent has a distinct system prompt and persona")
print("4. This pattern scales well as you add more domains (Sales, HR, Legal)")

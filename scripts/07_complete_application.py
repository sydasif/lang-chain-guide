# This script requires the 'langchain-community' and 'fastembed' packages.
# You can install them with:
# pip install langchain-community fastembed

import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
from langgraph.runtime import Runtime

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables


# 1. Define Tools
@tool
def get_order_status(order_id: str) -> str:
    """Get the status of a customer order."""
    orders = {
        "ORD001": "Shipped - arriving tomorrow",
        "ORD002": "Processing - will ship today",
        "ORD003": "Delivered on Nov 28",
    }
    return orders.get(order_id, "Order not found")


@tool
def initiate_return(order_id: str, reason: str = "Not specified") -> str:
    """Start a return process for an order."""
    return (
        f"Return initiated for order {order_id}. Reason: {reason}. "
        "Return label sent to email."
    )


@tool
def check_inventory(product_name: str) -> str:
    """Check if a product is in stock."""
    inventory = {
        "laptop": "In stock - 15 units available",
        "phone": "Low stock - 3 units remaining",
        "tablet": "Out of stock - restocking Dec 5",
    }
    return inventory.get(product_name.lower(), "Product not found")


# 2. Create Knowledge Base (RAG) - Custom Retriever Tool
knowledge_docs = [
    "Our return policy allows returns within 30 days of purchase for a full refund.",
    "Shipping is free for orders over $50. Standard shipping takes 3-5 business days.",
    "We offer 24/7 customer support via chat, email, and phone.",
    "All products come with a 1-year manufacturer warranty.",
    "You can track your order using the order number sent to your email.",
]

documents = [Document(page_content=doc) for doc in knowledge_docs]

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def company_policies(query: str) -> str:
    """Search company policies, shipping info, and customer service guidelines."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


# 3. Customer Service Middleware (Logging & Observability)
class CustomerServiceMiddleware(AgentMiddleware):
    """Middleware for logging customer interactions."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print(f"\n{'=' * 50}")
        print(
            f"🕐 {datetime.now().strftime('%H:%M:%S')} - Processing customer query..."
        )
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("✅ Response ready")
        return None


# 4. Initialize Agent
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

system_prompt = """You are a helpful customer service agent for an e-commerce company.

Your responsibilities:
- Help customers with order inquiries
- Process returns and exchanges
- Check product inventory
- Answer policy questions using the knowledge base
    - Summarize the relevant policy and stop.

Be friendly, professional, and efficient.
Always verify order IDs before processing requests.
Once you have the information or have performed the action,
answer the user immediately."""

tools = [get_order_status, initiate_return, check_inventory, company_policies]

# Create agent with middleware
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[CustomerServiceMiddleware()],
    system_prompt=system_prompt,
)


# 5. Run Customer Service Sessions
def handle_customer(customer_id: str, query: str):
    """Handle a customer service interaction."""
    # The middleware handles logging now!

    try:
        # Add recursion_limit to prevent infinite loops and save tokens
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 20},
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error processing request: {str(e)}"


# Example Interactions
print("\n" + "=" * 70)
print("CUSTOMER SERVICE AGENT - LIVE DEMO")
print("=" * 70)

# Session 1: Order Status
response1 = handle_customer(
    "CUST123", "Hi, I'd like to check the status of my order ORD001"
)
print(f"\n📞 Agent: {response1}")

# Session 2: Return Request
response2 = handle_customer(
    "CUST456", "I need to return order ORD002, the product doesn't fit"
)
print(f"\n📞 Agent: {response2}")

# Session 3: Policy Question
response3 = handle_customer("CUST789", "What's your return policy?")
print(f"\n📞 Agent: {response3}")

# Session 4: Inventory Check
response4 = handle_customer("CUST321", "Do you have laptops in stock?")
print(f"\n📞 Agent: {response4}")

print("\n" + "=" * 70)
print("SESSION COMPLETE")
print("=" * 70)

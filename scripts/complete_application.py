# This script requires the 'langchain-community' and 'fastembed' packages.
# You can install them with:
# pip install langchain-community fastembed

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")


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
def initiate_return(order_id: str, reason: str) -> str:
    """Start a return process for an order."""
    return f"Return initiated for order {order_id}. Reason: {reason}. Return label sent to email."


@tool
def check_inventory(product_name: str) -> str:
    """Check if a product is in stock."""
    inventory = {
        "laptop": "In stock - 15 units available",
        "phone": "Low stock - 3 units remaining",
        "tablet": "Out of stock - restocking Dec 5",
    }
    return inventory.get(product_name.lower(), "Product not found")


# 2. Create Knowledge Base (RAG)
knowledge_docs = [
    "Our return policy allows returns within 30 days of purchase for a full refund.",
    "Shipping is free for orders over $50. Standard shipping takes 3-5 business days.",
    "We offer 24/7 customer support via chat, email, and phone.",
    "All products come with a 1-year manufacturer warranty.",
    "You can track your order using the order number sent to your email.",
]

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(knowledge_docs)
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

retriever_tool = create_retriever_tool(
    retriever,
    "company_policies",
    "Search company policies, shipping info, and customer service guidelines.",
)


# 3. Customer Service Logging (replacing middleware)
class CustomerServiceLogger:
    def __init__(self):
        self.interactions = []

    def log_start(self):
        print(f"\n{'=' * 50}")
        print(
            f"🕐 {datetime.now().strftime('%H:%M:%S')} - Processing customer query..."
        )

    def log_tool_use(self, tool_name: str):
        print(f"🔧 Accessing: {tool_name}")

    def log_end(self, response):
        print("✅ Response ready")
        self.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "response_length": len(str(response)),
        })


# 4. Initialize Agent
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

system_prompt = """You are a helpful customer service agent for an e-commerce company.

Your responsibilities:
- Help customers with order inquiries
- Process returns and exchanges
- Check product inventory
- Answer policy questions using the knowledge base

Be friendly, professional, and efficient. Always verify order IDs before processing requests."""

tools = [get_order_status, initiate_return, check_inventory, retriever_tool]
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

logger = CustomerServiceLogger()


# 5. Run Customer Service Sessions
def handle_customer(customer_id: str, query: str):
    """Handle a customer service interaction."""
    # Log the start
    logger.log_start()

    # Create the message for the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    return result["messages"][-1].content


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

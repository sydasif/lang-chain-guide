#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 8: Complete Routing System
Combines classification, routing, and execution using LCEL
"""

import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("Network Automation Agent - Routing System")
print("=" * 60)


# ===== LAYER 1: CLASSIFIER =====
def classify_query(text: str) -> str:
    """Classify query as network task or general chat."""
    network_keywords = [
        "show",
        "display",
        "check",
        "interface",
        "ip",
        "bgp",
        "ospf",
        "device",
        "switch",
        "router",
        "config",
        "ping",
    ]

    text_lower = text.lower()
    is_network = any(keyword in text_lower for keyword in network_keywords)

    return "network_task" if is_network else "general"


# ===== LAYER 2: NETWORK TOOLS =====
@tool
def run_command(device: str, command: str) -> str:
    """
    Execute a network show command on a device.

    Args:
        device: Device hostname or IP
        command: Show command to execute

    Returns:
        Command output
    """
    # Safety check
    if not command.lower().startswith("show"):
        return "❌ Only 'show' commands are allowed."

    # Simulated output
    output = f"""
Output from {device}:
========================================
Interface          Status      Protocol    IP Address
GigE0/0            up          up          192.168.1.1
GigE0/1            up          up          10.0.0.1
Loopback0          up          up          192.168.100.1
========================================
"""
    return output.strip()


@tool
def ping_device(device: str) -> str:
    """
    Check device reachability.

    Args:
        device: Device hostname or IP

    Returns:
        Ping result
    """
    return f"✅ {device} is reachable (RTT: 5ms)"


# ===== LAYER 3: CHAINS =====

# General chat chain (no tools)
general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions clearly and concisely."),
    ("human", "{question}"),
])

general_chain = general_prompt | llm | (lambda x: x.content)

# Network task chain (with tools)
network_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a network automation assistant. "
        "Use the available tools to check device status when needed. "
        "Only use 'show' commands for safety.",
    ),
    ("human", "{question}"),
])

llm_with_tools = llm.bind_tools([run_command, ping_device])
network_chain = network_prompt | llm_with_tools


# ===== LAYER 4: ROUTING =====

# Classifier: adds category to input
classifier = RunnableLambda(lambda q: {"question": q, "category": classify_query(q)})


# Router: selects appropriate chain
def route_query(input_data):
    """Route to appropriate chain based on category."""
    category = input_data["category"]
    question = input_data["question"]

    if category == "network_task":
        return network_chain.invoke({"question": question})
    return general_chain.invoke({"question": question})


router = RunnableLambda(route_query)

# Complete pipeline: classifier → router
agent = classifier | router


# ===== TESTING =====
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Complete Routing System")
    print("=" * 60)

    test_queries = [
        "Show interface status on switch-core-01",
        "What's the capital of France?",
        "Check BGP neighbors on router-edge-02",
        "Tell me a joke",
        "Ping device 192.168.1.1",
        "How do I boil water?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"❓ Query: {query}")

        # Show classification
        category = classify_query(query)
        print(f"📊 Classified as: {category.upper()}")

        if category == "network_task":
            print("🔀 Routing to: Network Tool Chain")
        else:
            print("🔀 Routing to: General Chat Chain")

        print("-" * 60)

        # Get response
        result = agent.invoke(query)

        # Display result (handle different response types)
        if hasattr(result, "content"):
            print(f"💡 Response: {result.content[:200]}")
        elif hasattr(result, "tool_calls") and result.tool_calls:
            print(f"🔧 Tool calls detected: {len(result.tool_calls)}")
            for tc in result.tool_calls:
                print(f"   - Tool: {tc['name']}")
                print(f"   - Arguments: {tc['args']}")
        else:
            print(f"💡 Response: {str(result)[:200]}")

    print("\n" + "=" * 60)
    print("Routing System Architecture:")
    print("=" * 60)
    print("""
    User Query
        ↓
    Classifier (keyword matching)
        ↓
    Router (selects chain)
        ↓
    ┌───────┴────────┐
    ↓                ↓
General Chain    Network Chain
(chat only)      (with tools)
    |                |
    └────────┬───────┘
             ↓
        Response
    """)

    print("\n✨ Key Benefits:")
    print("   - Clean separation of concerns")
    print("   - Easy to test each layer")
    print("   - Efficient routing")
    print("   - Safe tool execution")
    print("=" * 60)

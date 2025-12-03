#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 7: Tools with LCEL
This script demonstrates using tools with LCEL (bind_tools) without full agent overhead
"""

import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("Tools with LCEL Demo")
print("=" * 60)


# Define tools
@tool
def add(x: int, y: int) -> int:
    """Add two numbers together."""
    print(f"🔧 TOOL CALLED: add({x}, {y})")
    return x + y


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    print(f"🔧 TOOL CALLED: multiply({x}, {y})")
    return x * y


@tool
def power(base: int, exponent: int) -> int:
    """Calculate base raised to the power of exponent."""
    print(f"🔧 TOOL CALLED: power({base}, {exponent})")
    return base**exponent


# Bind tools to the LLM
# This makes the LLM aware of the tools without creating a full agent
print("\n[Step 1] Binding tools to LLM...")
llm_with_tools = llm.bind_tools([add, multiply, power])
print("✓ Tools bound: add, multiply, power")

# Create a prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful math assistant. Use the available tools when needed.",
    ),
    ("human", "{question}"),
])

# Build the chain
chain = prompt | llm_with_tools

print("\n" + "=" * 60)
print("Testing Tool-Aware Chain")
print("=" * 60)

# Test questions
questions = [
    "What is 15 times 7?",
    "Calculate 25 plus 17",
    "What is 2 to the power of 8?",
    "If I have 5 apples and buy 3 more, then multiply them by 2, how many do I have?",
]

for question in questions:
    print(f"\n❓ Question: {question}")
    print("-" * 60)

    # Invoke the chain
    response = chain.invoke({"question": question})

    # Check if tools were called
    if response.tool_calls:
        print("🤖 AI decided to use tools:")
        for tool_call in response.tool_calls:
            print(f"   - Tool: {tool_call['name']}")
            print(f"   - Arguments: {tool_call['args']}")
    else:
        print(f"🤖 AI Response: {response.content}")

print("\n" + "=" * 60)
print("LCEL vs Agent Comparison:")
print("=" * 60)
print("\n✅ LCEL with bind_tools:")
print("   - Single LLM call")
print("   - Tool-aware but not tool-executing")
print("   - Fast and lightweight")
print("   - Good for predictable tool usage")

print("\n🤖 Full Agent:")
print("   - Multiple LLM calls (reasoning loop)")
print("   - Automatically executes tools")
print("   - Handles multi-turn conversations")
print("   - Good for complex orchestration")

print("\n💡 Key Insight:")
print("   bind_tools() gives you tool AWARENESS")
print("   Agents give you tool EXECUTION + ORCHESTRATION")
print("=" * 60)

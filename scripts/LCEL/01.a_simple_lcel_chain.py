#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 7: Simple LCEL Chain
This script demonstrates the most basic LCEL chain: prompt | llm | output_parser
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("Simple LCEL Chain Demo")
print("=" * 60)

# Create a prompt template
prompt = PromptTemplate.from_template("Write a short tip about: {topic}")

# Build the chain using the pipe operator
# This is the LCEL way: clean and composable
chain = prompt | llm | (lambda x: x.content)

# Test with different topics
topics = ["network automation", "Python programming", "DevOps best practices"]

for topic in topics:
    print(f"\n📌 Topic: {topic}")
    print("-" * 60)

    result = chain.invoke({"topic": topic})
    print(f"💡 Tip: {result}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)

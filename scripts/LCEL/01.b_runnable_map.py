#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 7: RunnableMap for Parallel Processing
This script demonstrates how to use RunnableMap to process data through multiple paths
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("RunnableMap Demo - Content Analyzer")
print("=" * 60)


# Define processing functions
def get_summary(text: str) -> str:
    """Generate a one-sentence summary."""
    prompt = f"Summarize this in ONE sentence: {text}"
    return llm.invoke(prompt).content


def extract_keywords(text: str) -> list:
    """Extract important keywords (simple approach)."""
    words = text.lower().split()
    # Simple keyword extraction - words longer than 6 characters
    keywords = [w for w in words if len(w) > 6]
    return keywords[:5]  # Return top 5


def analyze_sentiment(text: str) -> str:
    """Determine sentiment of the text."""
    prompt = (
        f"Is this text positive, negative, or neutral? "
        f"Answer with ONE word only: {text}"
    )
    return llm.invoke(prompt).content.strip()


# Create a parallel processing pipeline using RunnableMap
# All branches execute simultaneously!
analyzer = RunnableMap({
    "original_text": lambda x: x,
    "word_count": lambda x: len(x.split()),
    "character_count": lambda x: len(x),
    "summary": RunnableLambda(get_summary),
    "keywords": RunnableLambda(extract_keywords),
    "sentiment": RunnableLambda(analyze_sentiment),
})

# Test text
text = (
    "LangChain provides powerful abstractions for building sophisticated "
    "AI applications with minimal code. The LCEL pipeline approach makes "
    "it easy to compose complex workflows from simple building blocks."
)

print("\n📄 Input Text:")
print(f'"{text}"')
print("\n" + "=" * 60)
print("Analysis Results (Parallel Processing)")
print("=" * 60)

# Run the analysis
result = analyzer.invoke(text)

# Display results
print(f"\n📊 Original Text: {result['original_text'][:50]}...")
print(f"🔢 Word Count: {result['word_count']}")
print(f"🔢 Character Count: {result['character_count']}")
print(f"💡 Summary: {result['summary']}")
print(f"🔑 Keywords: {', '.join(result['keywords'])}")
print(f"😊 Sentiment: {result['sentiment']}")

print("\n" + "=" * 60)
print("Why This Is Powerful:")
print("- All analyses run in PARALLEL")
print("- Single invoke() call")
print("- Clean, readable code")
print("- Easy to add new analysis types")
print("=" * 60)

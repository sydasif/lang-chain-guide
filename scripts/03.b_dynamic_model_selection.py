import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Retrieve the GROQ API key
api_key = os.getenv("GROQ_API_KEY")


def analyze_query_complexity(query: str) -> str:
    """
    Analyze the complexity of a query to determine which model to use.
    In a real app, this could be a classifier or a smaller LLM.
    """
    query_lower = query.lower()

    # Keywords that suggest a complex reasoning task
    complex_keywords = [
        "explain",
        "compare",
        "analyze",
        "design",
        "architecture",
        "why",
        "how",
        "relationship",
        "difference",
        "code",
        "debug",
    ]

    # Check for length or keywords
    if len(query.split()) > 20 or any(
        keyword in query_lower for keyword in complex_keywords
    ):
        return "complex"
    return "simple"


def get_smart_model(query: str):
    """
    Select appropriate model based on query complexity.

    Strategy:
    - Simple queries -> Llama 3.1 8B (Fast, Cheap)
    - Complex queries -> Llama 3.3 70B (Powerful, More Expensive)
    """
    complexity = analyze_query_complexity(query)

    print(f"\n[ROUTER] Analyzing query: '{query[:50]}...'")
    print(f"[ROUTER] Complexity: {complexity.upper()}")

    if complexity == "simple":
        print("[ROUTER] Selection: Llama 3.1 8B (Fast Model)")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    print("[ROUTER] Selection: Llama 3.3 70B (Powerful Model)")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


# --- Demonstration ---

print("=" * 60)
print("SMART MODEL SELECTION DEMONSTRATION")
print("=" * 60)

test_queries = [
    "What is the capital of France?",  # Simple fact
    "Explain the difference between TCP and UDP and when to use each.",  # Complex explanation
    "Hello, how are you?",  # Simple greeting
    "Write a Python script to scrape a website and save data to CSV.",  # Complex coding task
]

for query in test_queries:
    print("-" * 60)

    # 1. Select the model dynamically
    start_time = time.time()
    model = get_smart_model(query)

    # 2. Invoke the selected model
    response = model.invoke(query)
    end_time = time.time()

    # 3. Print results
    print(f"AI Response: {response.content[:100]}...")
    print(f"Time taken: {end_time - start_time:.2f}s")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("1. Simple queries are routed to smaller, faster models")
print("2. Complex queries are routed to larger, more capable models")
print("3. This optimizes both COST and LATENCY")
print("4. The user gets the best experience for their specific need")

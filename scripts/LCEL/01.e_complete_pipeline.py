#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 7: Complete LCEL Pipeline
This script demonstrates a complete end-to-end pipeline combining:
- Classification
- Routing
- RAG
- Multiple chains
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableMap
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Complete LCEL Pipeline - Smart Q&A System")
print("=" * 60)

# ===== STEP 1: Build Knowledge Base =====
print("\n[Step 1] Building knowledge base...")
raw_documents = [
    "LangChain provides tools for building AI applications with LLMs.",
    "RAG combines retrieval with generation for better accuracy and grounding.",
    "LCEL is the LangChain Expression Language for building clean pipelines.",
    "Vector databases enable semantic search using embeddings.",
    "Embeddings are numerical representations that capture semantic meaning.",
]

documents = [Document(page_content=doc) for doc in raw_documents]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print(f"✓ Knowledge base ready with {len(docs)} chunks")


# ===== STEP 2: Create Helper Functions =====
def classify_question(question: str) -> str:
    """Classify if question needs knowledge base or can be answered generally."""
    knowledge_keywords = ["what is", "define", "explain", "how does", "tell me about"]
    is_knowledge = any(keyword in question.lower() for keyword in knowledge_keywords)
    return "knowledge" if is_knowledge else "general"


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


print("\n[Step 2] Setting up classifier and routing logic...")


# ===== STEP 3: Build the Two Paths =====

# Path A: General questions (no retrieval needed)
general_chain = (
    PromptTemplate.from_template(
        "Answer this question briefly and helpfully: {question}"
    )
    | llm
    | (lambda x: x.content)
)

# Path B: Knowledge-based questions (use RAG)
knowledge_chain = (
    RunnableMap({
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
    })
    | PromptTemplate.from_template(
        """Use the context below to answer the question accurately.

Context:
{context}

Question: {question}

Answer:"""
    )
    | llm
    | (lambda x: x.content)
)

print("✓ Created two chains: general_chain and knowledge_chain")


# ===== STEP 4: Build Classifier and Router =====

# Classifier: adds category to the input
classifier = RunnableLambda(lambda q: {"question": q, "category": classify_question(q)})

# Router: selects the appropriate chain based on category
router = RunnableBranch(
    # If category is "knowledge", use knowledge chain
    (lambda x: x["category"] == "knowledge", lambda x: knowledge_chain.invoke(x)),
    # Otherwise (default), use general chain
    lambda x: general_chain.invoke(x),
)

# Complete pipeline: classifier → router
complete_pipeline = classifier | router

print("✓ Pipeline ready: classifier → router → appropriate chain")


# ===== STEP 5: Test the Complete Pipeline =====
print("\n" + "=" * 60)
print("Testing Complete Pipeline")
print("=" * 60)

test_questions = [
    "What is LCEL?",
    "Write a haiku about programming",
    "Explain RAG in simple terms",
    "What's your favorite color?",
    "How do embeddings work?",
    "Tell me a joke about AI",
]

for question in test_questions:
    print(f"\n{'=' * 60}")
    print(f"❓ Question: {question}")

    # Classify to show routing decision
    category = classify_question(question)
    print(f"📊 Category: {category.upper()}")
    print(
        f"🔀 Route: {'Knowledge Chain (RAG)' if category == 'knowledge' else 'General Chain'}"
    )
    print("-" * 60)

    # Invoke the complete pipeline
    answer = complete_pipeline.invoke(question)

    print(f"💡 Answer: {answer}")

print("\n" + "=" * 60)
print("Pipeline Architecture Summary")
print("=" * 60)
print("""
Input Question
      ↓
  Classifier (determines intent)
      ↓
    Router (selects appropriate chain)
      ↓
   ┌──────┴──────┐
   ↓             ↓
General     Knowledge
 Chain        Chain
   |             |
   | (no RAG)    | (with RAG)
   |             |
   └──────┬──────┘
          ↓
     Final Answer
""")

print("\n✨ Key Benefits:")
print("   - Clean separation of concerns")
print("   - Easy to test each component")
print("   - Efficient (only retrieves when needed)")
print("   - Simple to extend (add new categories)")
print("   - Production-ready architecture")
print("=" * 60)

# This script demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline
# using LangChain, FAISS vector store, and local embeddings.
#
# Requirements:
# pip install langchain-community fastembed langchain-groq python-dotenv

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatGroq language model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("RAG Pipeline Demo - LangChain Zero-to-Hero Part 4")
print("=" * 60)

# Step 1: Load your documents
print("\n[Step 1] Loading documents...")
raw_documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings.",
]
documents = [Document(page_content=doc) for doc in raw_documents]
print(f"✓ Loaded {len(documents)} documents")

# Step 2: Split documents into chunks using RecursiveCharacterTextSplitter
print("\n[Step 2] Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # Maximum characters per chunk
    chunk_overlap=20,      # Overlap between chunks to maintain context
    add_start_index=True,  # Track position in original document
)
docs = text_splitter.split_documents(documents)
print(f"✓ Split into {len(docs)} chunks")

# Step 3: Create embeddings and vector store (using a free, local model)
print("\n[Step 3] Creating embeddings and building vector store...")
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
print(f"✓ Vector store created with {len(docs)} document chunks")

# Step 4: Create a retriever
print("\n[Step 4] Creating retriever...")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},  # Return top 2 most relevant documents
)
print("✓ Retriever configured (similarity search, k=2)")


# Step 5: Create a custom tool for retrieval
@tool
def knowledge_base(query: str) -> str:
    """Search for information about LangChain, RAG, and vector databases."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


# Step 6: Create agent with retriever tool
print("\n[Step 5] Creating RAG agent...")
tools = [knowledge_base]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant with access to a knowledge base. "
                  "When answering questions, use the knowledge base tool to find "
                  "relevant information before responding."
)
print("✓ Agent created with knowledge_base tool")

# Step 7: Ask questions that require retrieved knowledge
print("\n" + "=" * 60)
print("Testing RAG Agent")
print("=" * 60)

print("\n[Query 1] Using the agent to answer a question")
print("User: Explain what RAG is and how it works")
print("\nAgent is thinking and searching knowledge base...")

result = agent.invoke({
    "messages": [{"role": "user", "content": "Explain what RAG is and how it works"}]
})

print(f"\nAI: {result['messages'][-1].content}")

# Test direct retrieval
print("\n" + "=" * 60)
print("Testing Direct Retrieval")
print("=" * 60)

query = "What is FAISS?"
print(f"\n[Query 2] Direct retrieval test")
print(f"Query: '{query}'")

relevant_docs = retriever.invoke(query)

print(f"\nRetrieved {len(relevant_docs)} documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n{i}. {doc.page_content}")

print("\n" + "=" * 60)
print("RAG Pipeline Demo Complete!")
print("=" * 60)

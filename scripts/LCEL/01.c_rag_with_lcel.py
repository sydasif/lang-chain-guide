#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 7: RAG with LCEL
This script demonstrates building a RAG pipeline using LCEL (faster and simpler than agents)
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("=" * 60)
print("RAG Pipeline with LCEL")
print("=" * 60)

# Step 1: Create knowledge base
print("\n[Step 1] Building knowledge base...")
raw_documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings.",
    "LCEL (LangChain Expression Language) allows building pipelines with the pipe operator.",
]

documents = [Document(page_content=doc) for doc in raw_documents]

# Step 2: Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print(f"✓ Knowledge base ready with {len(docs)} chunks")


# Step 5: Helper function to format documents
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


# Step 6: Create the RAG prompt
rag_prompt = PromptTemplate.from_template(
    """Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""
)

# Step 7: Build the RAG chain using LCEL
print("\n[Step 2] Building RAG chain...")

# The complete RAG pipeline
rag_chain = (
    # First, create a map with question and retrieved context
    RunnableMap({
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
    })
    # Then pass to prompt template
    | rag_prompt
    # Then to LLM
    | llm
    # Finally extract content
    | (lambda x: x.content)
)

print("✓ RAG chain built successfully")

# Step 8: Test the RAG chain
print("\n" + "=" * 60)
print("Testing RAG Chain")
print("=" * 60)

questions = [
    "What is RAG and how does it work?",
    "What is FAISS used for?",
    "Explain LCEL in simple terms",
]

for question in questions:
    print(f"\n❓ Question: {question}")
    print("-" * 60)

    # Invoke the RAG chain
    answer = rag_chain.invoke({"question": question})

    print(f"💡 Answer: {answer}")

print("\n" + "=" * 60)
print("Why LCEL RAG is Better:")
print("- Single LLM call (faster)")
print("- Lower cost (fewer tokens)")
print("- Clean, readable pipeline")
print("- Easy to debug and test")
print("- Perfect for 'always retrieve' scenarios")
print("=" * 60)

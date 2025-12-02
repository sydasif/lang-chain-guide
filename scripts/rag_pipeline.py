# This script requires the 'langchain-community' and 'fastembed' packages.
# You can install them with:
# pip install langchain-community fastembed

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq language model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Step 1: Load your documents
from langchain_core.documents import Document
raw_documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings.",
]
documents = [Document(page_content=doc) for doc in raw_documents]

# Step 2: Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=200,  # Maximum characters per chunk
    chunk_overlap=20,  # Overlap between chunks to maintain context
)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store (using a free, local model)
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},  # Return top 2 most relevant documents
)

# Step 5: Create a custom tool for retrieval
@tool
def knowledge_base(query: str) -> str:
    """Search for information about LangChain, RAG, and vector databases."""
    docs = retriever.invoke(query)  # Use invoke instead of get_relevant_documents
    return "\n\n".join([doc.page_content for doc in docs])

# Step 6: Create agent with retriever tool
tools = [knowledge_base]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

# Step 7: Ask questions that require retrieved knowledge
print("User: Explain what RAG is and how it works")
result = agent.invoke({
    "messages": [{"role": "user", "content": "Explain what RAG is and how it works"}]
})

print(f"AI: {result['messages'][-1].content}")

# Test direct retrieval
print("\n--- Testing Retriever Directly ---")
query = "What is FAISS?"
relevant_docs = retriever.invoke(query)  # Use invoke instead of get_relevant_documents

print(f"Query: '{query}'")
print("Retrieved documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n{i}. {doc.page_content}")

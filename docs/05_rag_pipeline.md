# LangChain Zero-to-Hero Part 5: Giving Your AI a Photographic Memory (RAG Pipeline)

**Welcome to Part 5!** You've built an AI that can converse ([Part 1](./01_hello_world.md)), take action with tools ([Part 2](./02_tools_and_action.md)), adapt its behavior dynamically ([Part 3](./03_dynamic_behavior.md)), and remember conversations ([Part 4](./04_memory_and_context.md)). But there's still a critical limitation that makes your agent frustrating for real-world applications.

Try asking your current agent:

```bash
You: "What's our company's return policy?"
AI: "I don't have access to your specific company policies..."

You: "What did the CEO say in last week's all-hands meeting?"
AI: "I don't have information about recent meetings..."

You: "Where can I find documentation on configuring OSPF?"
AI: *Makes up an answer based on training data*
```

**The problem?** Your AI only knows what it learned during training. It can't access:

- Your company's internal documents
- Recent information (training data has a cutoff date)
- Private knowledge bases
- Proprietary documentation
- Real-time data sources

This is where **RAG (Retrieval-Augmented Generation)** transforms your agent from a conversational partner into a **domain expert**.

---

## What Is RAG? (The "Open Book Exam" for AI)

Remember taking tests in school? There were two types:

1. **Closed book**: You had to memorize everything
2. **Open book**: You could reference materials during the exam

LLMs are trained with a "closed book"—they only know what they memorized during training. RAG gives them an "open book"—your documents.

### The RAG Flow

```text
User Question
    ↓
AI analyzes question
    ↓
Searches your document database
    ↓
Retrieves relevant chunks
    ↓
AI reads retrieved content
    ↓
Generates answer based on YOUR documents
    ↓
Response with citations
```

This is fundamentally different from the tools you built in Part 2. Let's compare:

### Part 2 Tools (Action-Based)

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return api.fetch_weather(city)  # Calls external API
```

The AI executes an action and gets structured data back.

### Part 5 RAG (Knowledge-Based)

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search company documentation."""
    docs = retriever.invoke(query)  # Searches your documents
    return format_documents(docs)  # Returns relevant text
```

The AI searches through text and gets contextual information back.

**Both are tools, but they serve different purposes:**

- Tools = "Do something" (calculate, fetch data, execute commands)
- RAG = "Know something" (search, retrieve, reference)

---

## The Two Phases of RAG

RAG has two distinct phases that happen at different times:

### Phase 1: Indexing (One-Time Setup)

This happens **before** users start asking questions:

```text
Your Documents
    ↓
Load documents
    ↓
Split into chunks
    ↓
Convert to embeddings (vectors)
    ↓
Store in vector database
    ↓
[Ready for queries]
```

Think of this like building a library index—you do it once, then use it many times.

### Phase 2: Retrieval and Generation (Every Query)

This happens **every time** a user asks a question:

```text
User Query
    ↓
Convert query to embedding
    ↓
Search vector database for similar chunks
    ↓
Retrieve top K most relevant documents
    ↓
Pass documents + query to LLM
    ↓
LLM generates answer using retrieved context
```

Let's build both phases step by step.

---

## Phase 1: Building Your Knowledge Base

### Step 1 — Loading Documents

In production, you'll load documents from various sources. LangChain provides 160+ document loaders.

**Example: Loading from the Web**

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Only keep relevant content from HTML
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
print(f"Total characters: {len(docs[0].page_content)}")
```

**Other Common Loaders:**

```python
# PDF documents
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("company_policy.pdf")

# CSV files
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("customer_data.csv")

# Directory of files
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs", glob="**/*.md")
```

**For Our Tutorial: Simple In-Memory Documents**

To keep things simple, we'll start with hardcoded documents:

```python
from langchain_core.documents import Document

raw_documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings.",
]

documents = [Document(page_content=doc) for doc in raw_documents]
```

Each `Document` object has:

- `page_content`: The actual text
- `metadata`: Optional information (source, date, author, etc.)

---

### Step 2 — Splitting Documents Into Chunks

**Why split?** LLMs have context limits, and smaller chunks improve retrieval accuracy.

Imagine searching a 50-page manual for "how to reset password." You don't want the entire manual—you want the specific paragraph about password resets.

**The Challenge: Balancing Chunk Size**

- **Too large**: Irrelevant information dilutes the signal
- **Too small**: Loses important context
- **Just right**: Focused, self-contained pieces of information

LangChain's `RecursiveCharacterTextSplitter` is the recommended approach:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    add_start_index=True,   # Track position in original document
)

all_splits = text_splitter.split_documents(documents)
print(f"Split into {len(all_splits)} chunks")
```

### Understanding the Parameters

**chunk_size=1000**
Each chunk is roughly 1000 characters (~200-250 words). This is usually enough for a complete thought or concept.

**chunk_overlap=200**
The last 200 characters of each chunk overlap with the next chunk. This prevents splitting sentences or concepts awkwardly.

**Visualizing Overlap:**

```
Chunk 1: [=============================]
                              [=============================] Chunk 2
                                               [=============================] Chunk 3
         |<-- overlap -->|
```

**add_start_index=True**
Tracks where each chunk came from in the original document, useful for citations.

### Why RecursiveCharacterTextSplitter?

It tries to split on natural boundaries in this order:

1. Double newlines (paragraphs)
2. Single newlines (lines)
3. Spaces (words)
4. Characters (last resort)

This keeps semantic units together better than naive character splitting.

---

### Step 3 — Creating Embeddings

**What are embeddings?** Numerical representations of text that capture semantic meaning.

```
"The weather is sunny" → [0.2, -0.5, 0.8, ..., 0.3]  (384 numbers)
"It's a beautiful day"  → [0.3, -0.4, 0.7, ..., 0.4]  (similar numbers!)
"Database error 404"    → [-0.8, 0.9, -0.2, ..., 0.1] (very different!)
```

Similar meanings produce similar vectors. This enables semantic search—finding documents by meaning, not just keywords.

**Using FastEmbed (Free, Local, Fast):**

```python
from langchain_community.embeddings import FastEmbedEmbeddings

embeddings = FastEmbedEmbeddings()
```

**Why FastEmbed?**

- Runs locally (no API calls)
- Free (no usage costs)
- Fast (optimized for speed)
- Good quality (BAAI/bge-small-en-v1.5 model)

**Production Alternatives:**

```python
# OpenAI embeddings (high quality, costs money)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Cohere embeddings (good for multilingual)
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

---

### Step 4 — Storing in a Vector Database (FAISS)

**What is FAISS?** Facebook AI Similarity Search—a library for efficient similarity search in high-dimensional spaces.

**Why FAISS?**

- Blazingly fast (optimized C++ implementation)
- Scales to billions of vectors
- Runs locally (no external service)
- Free and open source
- Perfect for prototyping and production

**Creating the Vector Store:**

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(all_splits, embeddings)
```

**What just happened?**

1. Each document chunk was converted to a vector (embedding)
2. All vectors were indexed in FAISS for fast similarity search
3. You now have a searchable knowledge base

**Persisting Your Index (Important for Production!):**

```python
# Save the index to disk
vectorstore.save_local("my_knowledge_base")

# Later, load it back
vectorstore = FAISS.load_local(
    "my_knowledge_base",
    embeddings,
    allow_dangerous_deserialization=True  # Only if you trust the source
)
```

**Why persist?**

- Indexing is expensive (time and compute)
- You don't want to rebuild the index every time
- Enables incremental updates

**Alternative Vector Stores:**

```python
# Chroma (great for local development)
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(all_splits, embeddings)

# Pinecone (managed cloud service)
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(all_splits, embeddings)

# Weaviate (self-hosted or cloud)
from langchain_weaviate import WeaviateVectorStore
vectorstore = WeaviateVectorStore.from_documents(all_splits, embeddings)
```

---

## Phase 2: Retrieval and Generation

Now that your knowledge base is indexed, let's use it!

### Step 5 — Creating a Retriever

A retriever is an interface for fetching relevant documents:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)
```

### Understanding Retriever Parameters

**search_type Options:**

1. **"similarity"** (Default)
   - Returns documents most similar to the query
   - Fast and effective for most use cases

2. **"mmr"** (Maximal Marginal Relevance)
   - Balances relevance with diversity
   - Avoids returning duplicate information

   ```python
   retriever = vectorstore.as_retriever(
       search_type="mmr",
       search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
   )
   ```

3. **"similarity_score_threshold"**
   - Only returns documents above a similarity threshold
   - Useful for filtering low-quality matches

   ```python
   retriever = vectorstore.as_retriever(
       search_type="similarity_score_threshold",
       search_kwargs={"score_threshold": 0.7, "k": 5}
   )
   ```

**search_kwargs Parameters:**

- **k**: Number of documents to return (default: 4)
- **fetch_k**: Number of documents to fetch before filtering (for MMR)
- **lambda_mult**: Diversity vs relevance balance (0=diverse, 1=relevant)
- **score_threshold**: Minimum similarity score (0-1)

**Testing Your Retriever:**

```python
query = "What is FAISS?"
relevant_docs = retriever.invoke(query)

print(f"Query: '{query}'")
print(f"Retrieved {len(relevant_docs)} documents:\n")

for i, doc in enumerate(relevant_docs, 1):
    print(f"{i}. {doc.page_content}\n")
```

---

### Step 6 — Integrating RAG as a Tool (Simple Pattern)

Remember from Part 2 how tools work? Let's turn our retriever into a tool:

```python
from langchain.tools import tool

@tool
def knowledge_base(query: str) -> str:
    """Search for information about LangChain, RAG, and vector databases."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
```

**What makes a good RAG tool docstring?**

```python
# ❌ Too vague
"""Search the knowledge base."""

# ✅ Specific and descriptive
"""Search company documentation for information about policies, procedures,
and technical specifications. Use this when users ask about company-specific
information."""
```

The docstring tells the AI **when** to use the tool—be specific!

---

### Step 7 — Advanced RAG Tool Pattern (Production-Ready)

The official LangChain docs recommend a more sophisticated pattern using `response_format="content_and_artifact"`:

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)

    # Format for display to the LLM
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # Return both formatted text AND raw documents
    return serialized, retrieved_docs
```

**Why this pattern?**

1. **Content**: What the LLM sees (formatted text)
2. **Artifact**: Raw documents for downstream processing (citations, filtering, etc.)

This enables advanced features like:

- Automatic source citations
- Document filtering by metadata
- Multi-step reasoning with document context

---

### Step 8 — Building the RAG Agent

Now let's connect everything:

```python
from langchain.agents import create_agent
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

tools = [knowledge_base]  # or [retrieve_context] for advanced pattern

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant with access to a knowledge base. "
                  "When answering questions, use the knowledge base tool to find "
                  "relevant information before responding."
)
```

**The Magic of Agent-Based RAG:**

Unlike a simple RAG chain (query → retrieve → generate), an agent can:

1. **Decide when to search**: Not every question needs retrieval
2. **Make multiple searches**: Refine queries based on initial results
3. **Combine with other tools**: Mix retrieval with calculations, API calls, etc.

**Example: Multi-Step Reasoning**

```python
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

**What happens:**

1. Agent calls `retrieve_context("standard method for Task Decomposition")`
2. Reads the results
3. Calls `retrieve_context("common extensions of Task Decomposition")`
4. Synthesizes both results into a comprehensive answer

The agent **autonomously decided** to make two searches!

---

## RAG Agents vs RAG Chains

LangChain supports two RAG approaches:

### RAG Agent (What We Built)

```python
agent = create_agent(model=llm, tools=[retrieve_context])
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
```

**Pros:**

- Flexible (can make multiple retrievals)
- Combines with other tools
- Handles complex queries

**Cons:**

- Slower (multiple LLM calls)
- More expensive (more tokens)

**Use when:** Queries are complex or unpredictable

### RAG Chain (Simpler Alternative)

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the following context:\n\n{context}"),
    ("human", "{input}"),
])

# Create chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Use it
result = rag_chain.invoke({"input": "What is FAISS?"})
print(result["answer"])
```

**Pros:**

- Fast (single LLM call)
- Cheap (fewer tokens)
- Predictable

**Cons:**

- Fixed retrieval (always retrieves, can't adapt)
- No multi-step reasoning

**Use when:** Queries are simple and consistent

---

## Complete Working Example

Here's everything together:

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Step 1: Load documents
raw_documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings.",
]
documents = [Document(page_content=doc) for doc in raw_documents]

# Step 2: Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

# Step 5: Create RAG tool
@tool
def knowledge_base(query: str) -> str:
    """Search for information about LangChain, RAG, and vector databases."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Step 6: Create agent
agent = create_agent(
    model=llm,
    tools=[knowledge_base],
    system_prompt="You are a helpful assistant with access to a knowledge base."
)

# Step 7: Use the agent
print("User: Explain what RAG is and how it works")
result = agent.invoke({
    "messages": [{"role": "user", "content": "Explain what RAG is and how it works"}]
})
print(f"AI: {result['messages'][-1].content}")

# Test direct retrieval
print("\n--- Testing Retriever Directly ---")
query = "What is FAISS?"
relevant_docs = retriever.invoke(query)

print(f"Query: '{query}'")
print("Retrieved documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n{i}. {doc.page_content}")
```

---

## Real-World Use Cases

Now that you understand RAG, here's what becomes possible:

### Customer Support Knowledge Base

```python
# Load company documentation
loader = DirectoryLoader("./support_docs", glob="**/*.md")
docs = loader.load()

# Build searchable knowledge base
vectorstore = FAISS.from_documents(split_documents(docs), embeddings)

@tool
def search_support_docs(query: str) -> str:
    """Search company support documentation for answers to customer questions."""
    docs = vectorstore.similarity_search(query, k=3)
    return format_with_sources(docs)
```

### Network Documentation Assistant (DevNet Context)

```python
# Load Cisco documentation, RFCs, and internal runbooks
docs = load_network_documentation([
    "./cisco_docs",
    "./rfcs",
    "./runbooks"
])

@tool
def search_network_docs(query: str) -> str:
    """Search network device documentation, RFCs, and configuration guides."""
    results = vectorstore.similarity_search(query, k=5)
    return format_technical_docs(results)

# Now your agent can answer:
# "How do I configure OSPF on a Cisco router?"
# "What's the difference between EIGRP and OSPF?"
# "Show me the BGP configuration best practices"
```

### Code Documentation Search

```python
# Index your codebase documentation
loader = DirectoryLoader("./docs", glob="**/*.md")
code_docs = loader.load()

@tool
def search_code_docs(query: str) -> str:
    """Search project documentation and API references."""
    docs = vectorstore.similarity_search(query, k=4)
    return format_code_docs(docs)
```

### Policy and Compliance Checker

```python
@tool
def check_policy(query: str) -> str:
    """Search company policies, procedures, and compliance documents."""
    docs = vectorstore.similarity_search(query, k=3)

    # Include metadata for audit trail
    results = []
    for doc in docs:
        results.append(f"Policy: {doc.metadata['title']}\n"
                      f"Last Updated: {doc.metadata['date']}\n"
                      f"Content: {doc.page_content}\n")
    return "\n---\n".join(results)
```

---

## Production Considerations

### 1. Index Persistence and Updates

**Save your index:**

```python
# Initial build
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("./knowledge_base")

# Load in production
vectorstore = FAISS.load_local(
    "./knowledge_base",
    embeddings,
    allow_dangerous_deserialization=True
)
```

**Incremental updates:**

```python
# Add new documents without rebuilding
new_docs = load_new_documents()
vectorstore.add_documents(new_docs)
vectorstore.save_local("./knowledge_base")  # Persist changes
```

### 2. Handling Large Document Collections

**Problem:** Indexing 10,000 documents takes time and memory.

**Solution: Batch processing**

```python
from tqdm import tqdm

def build_large_index(documents, batch_size=100):
    vectorstore = None

    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        # Save periodically
        if i % 1000 == 0:
            vectorstore.save_local("./knowledge_base_checkpoint")

    return vectorstore
```

### 3. Metadata Filtering

**Add metadata to documents:**

```python
documents = [
    Document(
        page_content="OSPF configuration guide...",
        metadata={"source": "cisco_docs", "category": "routing", "date": "2024-01"}
    ),
    Document(
        page_content="BGP best practices...",
        metadata={"source": "rfc", "category": "routing", "date": "2023-12"}
    ),
]
```

**Filter during retrieval:**

```python
# Only search recent documents
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"date": {"$gte": "2024-01"}}
    }
)
```

### 4. Cost and Performance Optimization

**Reduce embedding costs:**

```python
# Use smaller, faster models for prototyping
embeddings = FastEmbedEmbeddings()  # Free, local

# Use high-quality models for production
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # $0.02/1M tokens
```

**Optimize chunk size:**

```python
# Smaller chunks = more precise but more expensive
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Larger chunks = less precise but cheaper
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
```

**Cache embeddings:**

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace="my_docs"
)
```

---

## Debugging and Best Practices

### Verifying Retrieval Quality

**Test your retriever before building the agent:**

```python
test_queries = [
    "What is RAG?",
    "How do embeddings work?",
    "What is FAISS used for?",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    docs = retriever.invoke(query)

    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
```

**Check if you're getting relevant results.** If not:

- Adjust chunk size
- Try different search types (MMR, threshold)
- Increase k (number of results)
- Improve document quality

### Handling "No Results Found"

```python
@tool
def knowledge_base(query: str) -> str:
    """Search the knowledge base."""
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the knowledge base."

    return "\n\n".join([doc.page_content for doc in docs])
```

### Monitoring Retrieval Relevance

```python
@tool(response_format="content_and_artifact")
def retrieve_with_scores(query: str):
    """Retrieve documents with similarity scores."""
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

    # Log low-quality retrievals
    for doc, score in docs_and_scores:
        if score < 0.5:  # Adjust threshold
            print(f"Warning: Low relevance score {score} for query: {query}")

    serialized = "\n\n".join(
        f"[Relevance: {score:.2f}] {doc.page_content}"
        for doc, score in docs_and_scores
    )

    return serialized, [doc for doc, _ in docs_and_scores]
```

### Common Pitfalls

**Pitfall 1: Chunks Too Large**

```python
# ❌ Chunks are entire documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)

# ✅ Chunks are focused paragraphs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

**Pitfall 2: No Overlap**

```python
# ❌ Sentences get cut in half
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# ✅ Context preserved across chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

**Pitfall 3: Forgetting to Persist**

```python
# ❌ Rebuild index every time (slow!)
vectorstore = FAISS.from_documents(docs, embeddings)

# ✅ Build once, load many times
if not os.path.exists("./knowledge_base"):
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("./knowledge_base")
else:
    vectorstore = FAISS.load_local("./knowledge_base", embeddings)
```

---

## Your Challenge: Build a Multi-Source RAG System

Before moving to Part 5, build this:

**Requirements:**

1. Load documents from at least 2 different sources (e.g., web + local files)
2. Add metadata to each document (source, category, date)
3. Create a retriever that filters by metadata
4. Build an agent with a RAG tool
5. Test with queries that should hit different sources

**Bonus Points:**

- Persist your vector store and load it on restart
- Implement source citations in responses
- Add a tool to list available document sources
- Create a "refresh index" function for new documents

**Example structure:**

```python
# Load from multiple sources
web_docs = WebBaseLoader(["https://example.com/docs"]).load()
local_docs = DirectoryLoader("./my_docs", glob="**/*.md").load()

# Add metadata
for doc in web_docs:
    doc.metadata["source"] = "web"
    doc.metadata["category"] = "public"

for doc in local_docs:
    doc.metadata["source"] = "local"
    doc.metadata["category"] = "internal"

# Build and test
all_docs = web_docs + local_docs
vectorstore = FAISS.from_documents(split_documents(all_docs), embeddings)

# Create filtered retrievers
public_retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"category": "public"}}
)

internal_retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"category": "internal"}}
)
```

This exercise will prepare you for real-world RAG systems with multiple knowledge sources.

---

## What You've Mastered

Incredible work! You've now learned:

- ✅ The two phases of RAG (indexing and retrieval)
- ✅ Document loaders for various sources
- ✅ Text splitting strategies with RecursiveCharacterTextSplitter
- ✅ Embeddings and semantic search
- ✅ FAISS vector store and persistence
- ✅ Retriever configuration and search types
- ✅ RAG tools (simple and advanced patterns)
- ✅ RAG agents vs RAG chains
- ✅ Production considerations and optimization
- ✅ Debugging and best practices

Your agent now has access to **domain-specific knowledge** without retraining!

---

## What's Next: Safety and Control

Your AI can now:

- Converse naturally (Part 1)
- Take actions with tools (Part 2)
- Remember context (Part 3)
- Access your knowledge base (Part 5)

But there's a critical problem: **your agent has no guardrails**.

What if it:

- Makes too many expensive API calls?
- Takes actions that need human approval?
- Generates inappropriate content?
- Exceeds rate limits?
- Costs spiral out of control?

**In Part 5**, we'll add the safety layer with **middleware**:

- Rate limiting and cost controls
- Human-in-the-loop approval
- Content filtering and moderation
- Logging and observability
- Error handling and retries
- Request validation

Your agent is about to gain structure, stability, and safety.

---

**Ready to make your agent production-safe?** Continue to [Part 5](./05_middleware_control.md) where we build the middleware layer that makes AI agents trustworthy for real-world deployment.

---

*Built something cool with RAG? Share your knowledge base implementation in the comments! I'd love to see what domains you're indexing.*

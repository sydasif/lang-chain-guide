# Part 4: The Open-Book Exam (RAG)

**Goal:** Build a Retrieval Augmented Generation (RAG) pipeline to let the AI "read" your documents.

## Why RAG?
You can't retrain a model every time your shipping policy changes. RAG lets you store knowledge in a database and fetch only the relevant pages when a user asks a question.

## The Pipeline (`scripts/rag_pipeline.py`)

### 1. Documents & Splitting
We break large text into small "chunks".
```python
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(raw_documents)
```

### 2. Embeddings & Vector Store
We convert text into numbers (vectors) and store them in FAISS (a fast local database).
```python
embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
```

### 3. The Retriever Tool
We turn the database search into a Tool the agent can use.
```python
@tool
def knowledge_base(query: str) -> str:
    """Search for info about our policies."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])
```

Now, when you ask "What is your return policy?", the Agent:
1. Calls the `knowledge_base` tool.
2. Reads the retrieved policy.
3. Synthesizes an answer.

**Next Step:** The agent is smart, but is it safe? [Part 5: Middleware](./05_middleware_control.md).
# Part 1: The Hello World of AI Agents

**Goal:** Get your environment set up and run your first AI interaction.

## Introduction
Building AI agents can feel overwhelming. You hear about RAG, Vectors, and Agents, but where do you start? We start at the very beginning: getting an LLM to say "Hello."

In this part, we use **Groq** (because it's fast and free for developers) and **LangChain** (the glue that holds everything together).

## The Setup
Security is step one. We never hardcode API keys. We use a `.env` file.

```python
# .env
GROQ_API_KEY=gsk_your_key_here
```

## The Basic Script
This script (`scripts/simple_model.py`) initializes a chat model and sends a single query.

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 1. Initialize the Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# 2. Ask a question
response = llm.invoke("What is the capital of France?")
print(response.content)
```

## Leveling Up: Streaming
Waiting for a full paragraph to generate feels slow. Modern UX demands streaming. Here is how we do it (`scripts/agent_streaming.py`):

```python
print("AI: ", end="", flush=True)

# Stream chunks as they arrive
for chunk in llm.stream("Why is the sky blue?"):
    print(chunk.content, end="", flush=True)
```

**Next Step:** Now that we can talk to the AI, let's teach it to *do* things. Proceed to [Part 2: Tools](./02_tools_and_action.md).
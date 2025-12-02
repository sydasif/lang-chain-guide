# LangChain Zero-to-Hero: Building Your First AI Agent (Hello World)

**Welcome to your AI development journey!** If you've been hearing about AI agents, RAG systems, and LangChain but don't know where to start, you're in the right place. This series will take you from complete beginner to building production-ready AI applications.

## Why This Tutorial Matters

The AI development landscape can feel overwhelming. Terms like "embeddings," "vector stores," "tool calling," and "retrieval-augmented generation" get thrown around constantly. But here's the truth: every sophisticated AI system starts with one simple action—**sending a prompt to a model and getting a response back**.

That's exactly what we're building today.

## What You'll Learn

By the end of this tutorial, you'll have:

- A secure development environment for AI projects
- Your first working LangChain script
- A streaming chatbot that responds in real-time
- The foundation for building advanced AI agents

We're using **Groq** with **Llama 3.3** (blazingly fast and free for developers) and **LangChain** (the industry-standard framework for building AI applications).

---

## Part 1: Secure Your API Keys (Don't Skip This!)

The first rule of AI development: **never hardcode your API keys**. This isn't just best practice—it's essential for security.

### Create Your Environment File

Open your terminal and create a `.env` file in your project root:

```bash
touch .env
```

Add your Groq API key to this file:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

**Why does this matter?**

- Keeps secrets out of your codebase
- Prevents accidental commits to GitHub
- Makes it easy to switch between development and production environments

LangChain automatically loads these variables using `python-dotenv`, so you don't need to write extra code.

---

## Part 2: Your First AI Interaction

Let's write the simplest possible AI script. This is the "Hello World" of LangChain development.

### The Code

Create a file called `simple_model.py`:

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

### Breaking It Down

Let's understand each piece:

**1. Loading Your Environment**

```python
from dotenv import load_dotenv
load_dotenv()
```

This line reads your `.env` file and makes your API key available to the script.

**2. Initializing the Model**

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
```

- `model`: Specifies which Llama variant to use
- `temperature`: Controls randomness (0 = deterministic, 1 = creative)

**3. Making the Request**

```python
response = llm.invoke("What is the capital of France?")
```

The `invoke()` method sends your prompt to the model and waits for the complete response.

**4. Displaying the Result**

```python
print(response.content)
```

Extracts the text content from the response object.

### Run It

```bash
python simple_model.py
```

You should see:

```
Paris
```

**Congratulations!** You've just built your first AI application.

---

## Part 3: Adding Real-Time Streaming

Static responses work, but modern AI feels more natural when it streams text as it's generated—just like ChatGPT. Let's upgrade our script.

### The Streaming Script

Create `agent_streaming.py`:

```python
import getpass
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    timeout=None,
    max_retries=2
)

print("AI: ", end="", flush=True)

for chunk in llm.stream("Why is the sky blue?"):
    print(chunk.content, end="", flush=True)

print()
```

### What Changed?

**1. API Key Fallback**

```python
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
```

If the environment variable isn't found, the script prompts you to enter it manually.

**2. Streaming Loop**

```python
for chunk in llm.stream("Why is the sky blue?"):
    print(chunk.content, end="", flush=True)
```

Instead of `invoke()`, we use `stream()`. This returns an iterator that yields text chunks as they're generated.

The `end=""` and `flush=True` parameters make the text appear smoothly without line breaks.

### Run It

```bash
python agent_streaming.py
```

You'll see the response appear word-by-word, creating that familiar ChatGPT-like experience.

---

## Part 4: Experiment and Explore

Now that you have a working setup, try customizing your script. Replace the question with:

**Creative Tasks:**

- "Write a haiku about programming"
- "Create a short story about a robot learning to paint"

**Technical Explanations:**

- "Explain how neural networks work in simple terms"
- "What's the difference between supervised and unsupervised learning?"

**Practical Applications:**

- "Summarize the benefits of using LangChain"
- "Write a professional email requesting a meeting"

Each call behaves like a conversation with an intelligent assistant.

---

## Understanding Temperature

The `temperature` parameter deserves special attention:

- **0.0**: Deterministic, consistent responses (ideal for factual queries)
- **0.7**: Balanced creativity and consistency (great for general use)
- **1.0+**: Highly creative, unpredictable (useful for brainstorming)

Try running the same prompt with different temperatures to see how responses change!

---

## Common Issues and Solutions

### "API Key Not Found"

Make sure your `.env` file is in the same directory as your script and formatted correctly:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### "Module Not Found"

Install dependencies:

```bash
pip install langchain-groq python-dotenv
```

### Slow Responses

Groq is typically very fast. If you experience slowness:

- Check your internet connection
- Verify your API key has remaining credits
- Try a different model variant

---

## What You've Accomplished

In this tutorial, you've:

1. ✅ Set up a secure development environment
2. ✅ Made your first API call to an LLM
3. ✅ Implemented real-time streaming responses
4. ✅ Built the foundation for advanced AI applications

This might seem simple, but you've just created the core component that powers every AI agent, chatbot, and automation system.

---

## What's Next: The Power of Tools

Right now, your AI can only *talk*. It can answer questions and generate text, but it can't *do* anything in the real world.

**In Part 2**, we'll unlock the true power of LangChain by teaching your AI to use **Tools**—Python functions that let it:

- Search the web for current information
- Perform calculations
- Interact with APIs and databases
- Execute custom business logic

This is where your chatbot transforms from a conversational partner into an autonomous agent that can take action.

---

## Your Challenge

Before moving to Part 2, try this exercise:

**Build a "Joke Generator"** that:

1. Asks the user what topic they want a joke about
2. Uses LangChain to generate a joke
3. Streams the response in real-time

This will reinforce what you've learned and prepare you for more complex patterns.

---

**Ready to give your AI superpowers?** Continue to [Part 2: Tools and Actions](./02_tools_and_action.md) where we'll build an agent that can actually interact with the world.

---

*Have questions or want to share what you built? Drop a comment below! I read and respond to every one.*

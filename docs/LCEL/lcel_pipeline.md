# Building Clean AI Workflows (LCEL Pipelines)

**Welcome to Part 7!** You've come a long way. In [Part 1](./01_hello_world.md), you learned to talk to LLMs. In [Part 2](./02_tools_and_action.md), you gave your AI superpowers with tools. In [Part 3](./03_dynamic_behavior.md), you made agents adaptive and intelligent. In [Part 4](./04_memory_and_context.md), you gave them memory. In [Part 5](./05_rag_pipeline.md), you connected them to knowledge bases. In [Part 6](./06_middleware_control.md), you added control layers.

But let's be honest—**your agent code is probably getting messy**.

## The Problem with Complex Agent Systems

Try building something like this the traditional way:

- User asks a question
- Check if it's simple or complex
- Route to different models based on complexity
- Retrieve context from your knowledge base if needed
- Apply rate limiting
- Add logging middleware
- Stream the response
- Handle errors gracefully

**Using only agent wrappers and custom code, you'd write hundreds of lines**. And worse? It becomes hard to test, debug, and maintain.

There's a better way.

---

## What is LCEL? (The Pipeline Revolution)

**LCEL** stands for **LangChain Expression Language**. Think of it as Unix pipes for AI workflows.

In Unix, you chain simple commands:

```bash
cat file.txt | grep "error" | sort | uniq
```

LCEL lets you do the same with AI components:

```python
prompt | llm | output_parser
```

**Why does this matter?**

- **Cleaner code**: One line instead of ten
- **Composable**: Mix and match building blocks
- **Streaming built-in**: No special handling needed
- **Async support**: Parallel execution for free
- **Type-safe**: Catch errors before runtime
- **Testable**: Each component can be tested independently

This is the **modern way** to build production AI systems with LangChain.

---

## What You'll Learn

By the end of this tutorial, you'll master:

1. **Basic LCEL chains** — Connecting prompts, models, and parsers
2. **RunnableMap** — Branching logic and parallel processing
3. **RAG with LCEL** — Retrieval pipelines without agents
4. **Tool integration** — Using tools without agent overhead
5. **Streaming** — Real-time responses in pipelines
6. **Complete workflows** — Building production-ready systems

Let's dive in.

---

## Part 1: Your First LCEL Chain (The Foundation)

Let's start with the simplest possible pipeline: turn a user's topic into a helpful tip.

### The Old Way (Verbose)

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = PromptTemplate.from_template("Write a short tip about: {topic}")

# Traditional approach (verbose)
formatted_prompt = prompt.format(topic="network automation")
response = llm.invoke(formatted_prompt)
final_output = response.content

print(final_output)
```

**That's 3 steps with 3 intermediate variables**. It works, but it's clunky.

### The LCEL Way (Clean)

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = PromptTemplate.from_template("Write a short tip about: {topic}")

# LCEL approach (clean)
chain = prompt | llm | (lambda x: x.content)

result = chain.invoke({"topic": "network automation"})
print(result)
```

### Breaking It Down

**The Pipe Operator (`|`)**

```python
chain = prompt | llm | (lambda x: x.content)
```

This creates a sequence where:

1. `prompt` receives `{"topic": "network automation"}`
2. Formats it into a complete prompt
3. Passes the prompt to `llm`
4. LLM generates a response
5. Lambda function extracts the text content

**Think of it like water flowing through pipes**:

```
Input → Prompt Template → LLM → Text Extractor → Output
```

Each component transforms the data and passes it forward.

### Run It

Create `scripts/07.a_simple_lcel_chain.py` and run:

```bash
uv run scripts/07.a_simple_lcel_chain.py
```

You'll see a clean tip about network automation, generated with just one line of logic!

---

## Part 2: Understanding Runnables (The Building Blocks)

LCEL chains are made of **Runnables**—objects that can be invoked, streamed, or batched.

### Key Runnable Types

**1. RunnableLambda — Custom Python Functions**

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

chain = RunnableLambda(uppercase)
result = chain.invoke("hello world")
print(result)  # OUTPUT: HELLO WORLD
```

**2. RunnableMap — Parallel Processing**

```python
from langchain_core.runnables import RunnableMap

# Process input through multiple paths simultaneously
chain = RunnableMap({
    "uppercase": lambda x: x.upper(),
    "lowercase": lambda x: x.lower(),
    "length": lambda x: len(x),
})

result = chain.invoke("Hello World")
print(result)
# {'uppercase': 'HELLO WORLD', 'lowercase': 'hello world', 'length': 11}
```

**3. RunnablePassthrough — Pass Data Forward**

```python
from langchain_core.runnables import RunnablePassthrough

# Useful for preserving original input alongside transformations
chain = RunnableMap({
    "original": RunnablePassthrough(),
    "processed": lambda x: x.upper(),
})

result = chain.invoke("hello")
print(result)
# {'original': 'hello', 'processed': 'HELLO'}
```

**4. RunnableSequence — Step-by-Step Execution**

```python
# This is what the pipe operator creates automatically
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(first=prompt, middle=llm, last=output_parser)
# Equivalent to: prompt | llm | output_parser
```

---

## Part 3: Branching Logic with RunnableMap

Real applications need to process data in multiple ways. RunnableMap makes this elegant.

### The Scenario

You're building a content analyzer that needs to:

- Generate a summary
- Extract keywords
- Count words
- Determine sentiment

All from the same input text.

### The LCEL Solution

Create `scripts/07.b_runnable_map.py`:

```python
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="llama-3.1-8b-instant")

# Define each processing step
def get_summary(text):
    prompt = f"Summarize in one sentence: {text}"
    return llm.invoke(prompt).content

def extract_keywords(text):
    words = text.lower().split()
    # Simple keyword extraction (in production, use NLP)
    return [w for w in words if len(w) > 6][:5]

def analyze_sentiment(text):
    prompt = f"Is this text positive, negative, or neutral? One word answer: {text}"
    return llm.invoke(prompt).content.strip()

# Create parallel processing pipeline
analyzer = RunnableMap({
    "original_text": lambda x: x,
    "word_count": lambda x: len(x.split()),
    "summary": RunnableLambda(get_summary),
    "keywords": RunnableLambda(extract_keywords),
    "sentiment": RunnableLambda(analyze_sentiment),
})

# Use it
text = "LangChain provides powerful abstractions for building sophisticated AI applications with minimal code."

result = analyzer.invoke(text)

print("Analysis Results:")
print(f"📝 Word Count: {result['word_count']}")
print(f"💡 Summary: {result['summary']}")
print(f"🔑 Keywords: {', '.join(result['keywords'])}")
print(f"😊 Sentiment: {result['sentiment']}")
```

### Why This Pattern is Powerful

**Without LCEL (ugly approach):**

```python
# Sequential processing - slow!
summary = get_summary(text)
keywords = extract_keywords(text)
sentiment = analyze_sentiment(text)
word_count = len(text.split())

result = {
    "summary": summary,
    "keywords": keywords,
    "sentiment": sentiment,
    "word_count": word_count
}
```

**With LCEL (elegant and parallel):**

```python
result = analyzer.invoke(text)
```

The RunnableMap **automatically runs independent steps in parallel**, making your pipeline faster!

---

## Part 4: RAG Pipeline Without Agents

Remember [Part 5](./05_rag_pipeline.md) where you built RAG agents? LCEL lets you build **simpler, faster RAG systems** when you don't need full agent autonomy.

### The Problem with RAG Agents

From Part 5, your RAG agent looked like this:

```python
@tool
def knowledge_base(query: str) -> str:
    """Search for information."""
    docs = retriever.invoke(query)
    return format_docs(docs)

agent = create_agent(model=llm, tools=[knowledge_base])
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

**Issues:**

- Multiple LLM calls (expensive)
- Tool selection overhead (slower)
- More complex debugging
- Overkill for "always retrieve" scenarios

### The LCEL Solution

Create `scripts/07.c_rag_with_lcel.py`:

```python
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Step 1: Create knowledge base (same as Part 5)
raw_documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings.",
]

documents = [Document(page_content=doc) for doc in raw_documents]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 2: Helper function to format documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Step 3: Create the RAG pipeline using LCEL
rag_prompt = PromptTemplate.from_template(
    """Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""
)

# The complete RAG chain
rag_chain = (
    # First, create a map with question and retrieved docs
    RunnableMap({
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
    })
    # Then pass to prompt
    | rag_prompt
    # Then to LLM
    | llm
    # Finally extract content
    | (lambda x: x.content)
)

# Use it
question = "What is RAG and how does it work?"
answer = rag_chain.invoke({"question": question})

print(f"Question: {question}")
print(f"Answer: {answer}")
```

### Breaking Down the RAG Chain

Let's trace the data flow step by step:

**Input:**

```python
{"question": "What is RAG?"}
```

**Step 1: RunnableMap**

```python
RunnableMap({
    "context": lambda x: format_docs(retriever.invoke(x["question"])),
    "question": lambda x: x["question"],
})
```

Output:

```python
{
    "context": "RAG stands for...\n\nVector databases store...",
    "question": "What is RAG?"
}
```

**Step 2: Prompt Template**

```python
| rag_prompt
```

Output:

```
Use the context below to answer the question.

Context:
RAG stands for...

Question: What is RAG?

Answer:
```

**Step 3: LLM**

```python
| llm
```

Output: `AIMessage(content="RAG is a technique that...")`

**Step 4: Content Extractor**

```python
| (lambda x: x.content)
```

Output: `"RAG is a technique that combines retrieval with generation..."`

### When to Use RAG Chains vs RAG Agents

**Use RAG Chain (LCEL) when:**

- Every query needs retrieval (e.g., customer support docs)
- Speed matters (single LLM call)
- Cost matters (fewer tokens)
- Simple, predictable behavior is desired

**Use RAG Agent (Part 5 approach) when:**

- Queries might not need retrieval
- You have multiple tools to choose from
- Multi-step reasoning is required
- You need to retry or refine searches

---

## Part 5: Using Tools in LCEL Chains

You can bind tools directly to models without creating agents.

### The Scenario

You're building a calculator assistant that needs:

- AI to understand natural language
- Tools to perform actual calculations
- Clean responses

### The Agent Way (heavyweight)

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

agent = create_agent(model=llm, tools=[add])
result = agent.invoke({"messages": [{"role": "user", "content": "What is 5 + 3?"}]})
```

This works but involves:

- Agent reasoning loop
- Tool selection logic
- Multiple LLM calls
- Complex message handling

### The LCEL Way (lightweight)

Create `scripts/07.d_tools_with_lcel.py`:

```python
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Define tools
@tool
def add(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y

@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y

# Bind tools directly to the model
llm_with_tools = llm.bind_tools([add, multiply])

# Create a simple chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math assistant. Use the available tools when needed."),
    ("human", "{question}")
])

chain = prompt | llm_with_tools

# Use it
response = chain.invoke({"question": "What is 15 times 7?"})

print("AI Response:")
print(response)

# The model will decide to call the multiply tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"\nTool Called: {tool_call['name']}")
    print(f"Arguments: {tool_call['args']}")
```

### Understanding bind_tools()

```python
llm_with_tools = llm.bind_tools([add, multiply])
```

This tells the LLM:

- "These tools exist"
- "You can call them if needed"
- "Here's their schema"

**But it doesn't:**

- Create an agent loop
- Handle tool execution automatically
- Manage conversation history

You get **tool awareness** without **agent overhead**.

### When to Use Tools in LCEL

**Good for:**

- Single-turn tool calls
- Predictable tool usage
- Performance-critical applications
- Simple tool logic

**Not good for:**

- Multi-turn conversations
- Complex tool orchestration
- Retry logic
- State management

---

## Part 6: Streaming in LCEL (Real-Time Responses)

One of LCEL's killer features: **streaming is automatic**.

### Basic Streaming

Any chain can stream:

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = PromptTemplate.from_template("Write a paragraph about: {topic}")

chain = prompt | llm

# Stream the response
for chunk in chain.stream({"topic": "network automation"}):
    print(chunk.content, end="", flush=True)
print()
```

### Streaming with Multiple Steps

Even complex chains can stream:

```python
from langchain_core.runnables import RunnableMap

chain = (
    RunnableMap({
        "topic": lambda x: x,
        "uppercase_topic": lambda x: x.upper(),
    })
    | prompt
    | llm
)

for chunk in chain.stream("python programming"):
    print(chunk.content, end="", flush=True)
```

**The magic?** Intermediate steps run, but only the final LLM output streams to you.

### Streaming Events (Advanced)

Want to see everything happening inside the chain?

```python
async for event in chain.astream_events({"topic": "AI"}, version="v1"):
    if event["event"] == "on_llm_stream":
        print(event["data"]["chunk"].content, end="")
```

This lets you:

- Debug complex chains
- Show progress to users
- Track performance metrics

---

## Part 7: Complete End-to-End Pipeline

Let's combine everything into a production-ready system.

### The Scenario

You're building a **Smart Question-Answer System** that:

1. Classifies questions (general vs knowledge-based)
2. Routes to appropriate handler
3. Retrieves context for knowledge questions
4. Generates answers
5. Formats output beautifully

### The Complete Implementation

Create `scripts/07.e_complete_pipeline.py`:

```python
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableMap
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize
llm = ChatGroq(model="llama-3.1-8b-instant")

# Step 1: Build knowledge base
raw_documents = [
    "LangChain provides tools for building AI applications.",
    "RAG combines retrieval with generation for better accuracy.",
    "LCEL is the LangChain Expression Language for building pipelines.",
    "Vector databases enable semantic search using embeddings.",
]

documents = [Document(page_content=doc) for doc in raw_documents]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 2: Classifier function
def classify_question(question: str) -> str:
    """Classify if question needs knowledge base."""
    keywords = ["what is", "define", "explain", "how does", "tell me about"]
    return "knowledge" if any(k in question.lower() for k in keywords) else "general"

# Step 3: Format documents helper
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Step 4: Build the two paths
# Path A: General questions
general_chain = (
    PromptTemplate.from_template("Answer this question briefly: {question}")
    | llm
    | (lambda x: x.content)
)

# Path B: Knowledge-based questions
knowledge_chain = (
    RunnableMap({
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
    })
    | PromptTemplate.from_template(
        "Use this context to answer:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    | llm
    | (lambda x: x.content)
)

# Step 5: Build the classifier and router
classifier = RunnableLambda(
    lambda q: {"question": q, "category": classify_question(q)}
)

router = RunnableBranch(
    # If category is "knowledge", use knowledge chain
    (lambda x: x["category"] == "knowledge", lambda x: knowledge_chain.invoke(x)),
    # Otherwise, use general chain
    lambda x: general_chain.invoke(x)
)

# Step 6: Complete pipeline
complete_pipeline = classifier | router

# Test it
print("=" * 60)
print("Smart Question-Answer System")
print("=" * 60)

questions = [
    "What is LCEL?",
    "Write a haiku about programming",
    "Explain RAG in simple terms",
    "What's your favorite color?",
]

for question in questions:
    print(f"\n❓ Question: {question}")
    category = classify_question(question)
    print(f"📊 Category: {category}")

    answer = complete_pipeline.invoke(question)
    print(f"💡 Answer: {answer}")
    print("-" * 60)
```

### Breaking Down the Complete Pipeline

Let's trace how a question flows through:

**Input:** `"What is RAG?"`

**Step 1: Classifier**

```python
classifier = RunnableLambda(
    lambda q: {"question": q, "category": classify_question(q)}
)
```

Output: `{"question": "What is RAG?", "category": "knowledge"}`

**Step 2: Router**

```python
router = RunnableBranch(
    (lambda x: x["category"] == "knowledge", lambda x: knowledge_chain.invoke(x)),
    lambda x: general_chain.invoke(x)
)
```

Checks category → Routes to `knowledge_chain`

**Step 3: Knowledge Chain Execution**

```python
# Retrieves docs, formats prompt, calls LLM
knowledge_chain.invoke({"question": "What is RAG?"})
```

Output: `"RAG stands for Retrieval Augmented Generation..."`

### Why This Design Works

**1. Separation of Concerns**

- Classifier: Decides intent
- Router: Picks the right path
- Chains: Execute specialized logic

**2. Easy to Test**
Each component can be tested independently:

```python
# Test classifier
assert classify_question("What is RAG?") == "knowledge"
assert classify_question("Hello") == "general"

# Test retrieval
docs = retriever.invoke("RAG")
assert len(docs) > 0

# Test general chain
result = general_chain.invoke({"question": "Hi"})
assert len(result) > 0
```

**3. Easy to Extend**

Want to add a new category?

```python
# Add new classifier logic
def classify_question(question: str) -> str:
    if "calculate" in question.lower():
        return "math"
    # ... existing logic

# Add new chain
math_chain = prompt | llm_with_calculator_tools | output_parser

# Update router
router = RunnableBranch(
    (lambda x: x["category"] == "knowledge", knowledge_chain),
    (lambda x: x["category"] == "math", math_chain),
    general_chain  # default
)
```

---

## Production Best Practices

### 1. Error Handling in Chains

LCEL chains can fail at any step. Add graceful error handling:

```python
from langchain_core.runnables import RunnableLambda

def safe_retrieval(query):
    try:
        return retriever.invoke(query)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return []

safe_chain = (
    RunnableLambda(safe_retrieval)
    | RunnableLambda(format_docs)
    | rag_prompt
    | llm
)
```

### 2. Logging for Observability

Add logging to track what's happening:

```python
def log_and_pass(step_name):
    def logger(x):
        print(f"[{step_name}] Input: {str(x)[:100]}...")
        return x
    return RunnableLambda(logger)

chain = (
    log_and_pass("Classifier")
    | classifier
    | log_and_pass("Router")
    | router
    | log_and_pass("Final Output")
)
```

### 3. Caching Expensive Operations

Use FAISS persistence to avoid rebuilding:

```python
# Build once
vectorstore.save_local("./knowledge_base")

# Load in production
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.load_local(
    "./knowledge_base",
    embeddings,
    allow_dangerous_deserialization=True
)
```

### 4. Async Execution for Speed

Convert chains to async for better performance:

```python
async def process_questions(questions):
    tasks = [chain.ainvoke({"question": q}) for q in questions]
    return await asyncio.gather(*tasks)

# Process 100 questions in parallel
answers = await process_questions(question_list)
```

---

## Real-World Use Cases

### Customer Support Bot

```python
# Classifier routes to specialized chains
support_pipeline = (
    classifier
    | RunnableBranch(
        (is_technical_question, tech_support_chain),
        (is_billing_question, billing_chain),
        (is_general_question, general_support_chain),
    )
)
```

### Document Analysis Pipeline

```python
# Parallel analysis of uploaded documents
doc_analyzer = RunnableMap({
    "summary": summary_chain,
    "keywords": keyword_extraction_chain,
    "sentiment": sentiment_chain,
    "entities": entity_recognition_chain,
})
```

### Multi-Language Translation

```python
# Detect language, then translate
translation_pipeline = (
    detect_language
    | RunnableBranch(
        (lambda x: x["lang"] == "es", spanish_translator),
        (lambda x: x["lang"] == "fr", french_translator),
        english_passthrough,
    )
)
```

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Forgetting to Extract Content

```python
# ❌ Wrong - returns AIMessage object
chain = prompt | llm

# ✅ Correct - returns string
chain = prompt | llm | (lambda x: x.content)
```

### Pitfall 2: Incorrect Input Format

```python
# ❌ Wrong - string input to prompt expecting dict
prompt = PromptTemplate.from_template("Hello {name}")
chain = prompt | llm
result = chain.invoke("Alice")  # ERROR!

# ✅ Correct - dictionary input
result = chain.invoke({"name": "Alice"})
```

### Pitfall 3: Not Handling Missing Keys

```python
# ✅ Safe approach
chain = (
    RunnableMap({
        "context": lambda x: x.get("context", ""),
        "question": lambda x: x.get("question", ""),
    })
    | prompt
    | llm
)
```

---

## What You've Mastered

Congratulations! You now understand:

- ✅ The pipe operator (`|`) for chaining components
- ✅ RunnableMap for parallel processing
- ✅ Building RAG pipelines without agents
- ✅ Integrating tools with `bind_tools()`
- ✅ Streaming responses automatically
- ✅ Building complete production pipelines
- ✅ Error handling and best practices

**LCEL is your secret weapon** for building clean, testable, production-ready AI applications.

---

## Your Challenge: Build a Smart Content Processor

Before moving to Part 8, build a pipeline that:

1. **Accepts** a blog post URL
2. **Extracts** the content
3. **Generates** three things in parallel:
   - A 2-sentence summary
   - 5 key takeaways
   - A suggested social media post
4. **Formats** the output beautifully

**Bonus:** Add error handling for invalid URLs and stream the summary generation.

This will test your understanding of RunnableMap, error handling, and streaming!

---

## What's Next: The Final Project

You've learned all the building blocks:

- Models and prompts ([Part 1](./01_hello_world.md))
- Tools ([Part 2](./02_tools_and_action.md))
- Dynamic behavior ([Part 3](./03_dynamic_behavior.md))
- Memory ([Part 4](./04_memory_and_context.md))
- RAG ([Part 5](./05_rag_pipeline.md))
- Middleware ([Part 6](./06_middleware_control.md))
- **Pipelines (Part 7)** ✅

**In [Part 8](./08_final_project.md)**, we bring it all together into a **complete production system**: the Network Automation Agent.

You'll see how all these concepts combine into:

- A real-world architecture
- Safe tool execution with guardrails
- Intelligent routing
- Production deployment
- Full logging and observability

This is where theory becomes practice.

---

**Ready to build the complete system?** Continue to [Part 8: Building Your Production-Ready Network Automation Agent](./08_final_project.md) where everything comes together.

---

*Questions about LCEL or want to share your pipeline creations? I'd love to see what you build!*

# **LangChain Zero-to-Hero – Part 6: Building a Real AI Agent (The Capstone Project)**

You’ve reached the final chapter.
Across the previous parts, you learned how to:

* Run an LLM
* Stream responses
* Give your agent tools
* Add memory
* Build a RAG pipeline
* Add safety through middleware

Now it’s time to **combine everything** into a single, real-world AI application.

This capstone project mirrors an industry-grade workflow:

> **Build a complete Customer Service Agent**
> that can answer questions, check orders, process returns, look up policies, and maintain logs.

Your repository contains the full implementation inside
`scripts/complete_application.py`.

This chapter breaks the design down into clear steps so you understand *how it all works together.*

---

## **The Goal**

A Customer Service Agent should be able to:

1. Check order status
2. Process returns
3. Search policy documents
4. Check product inventory
5. Provide friendly conversation
6. Log its own activity
7. Run safely and predictably

We will walk through each major component.

---

## **1. Define the Tools (Agent Capabilities)**

These are the “hands” of the agent:
Python functions wrapped with `@tool`.

### **Check order status**

```python
@tool
def get_order_status(order_id: str) -> str:
    orders = {
        "ORD001": "Shipped - arriving tomorrow",
        "ORD002": "Processing - will ship today",
        "ORD003": "Delivered on Nov 28",
    }
    return orders.get(order_id, "Order not found")
```

### **Process a return**

```python
@tool
def initiate_return(order_id: str, reason: str = "Not specified") -> str:
    return f"Return initiated for order {order_id}. Reason: {reason}. Return label sent to email."
```

### **Check inventory**

```python
@tool
def check_inventory(product_name: str) -> str:
    inventory = {
        "laptop": "In stock - 15 units available",
        "phone": "Low stock - 3 units remaining",
        "tablet": "Out of stock - restocking Dec 5",
    }
    return inventory.get(product_name.lower(), "Product not found")
```

These tools simulate a real e-commerce backend.

---

## **2. Build a Policy Knowledge Base (RAG)**

Policies change frequently.
You don’t want to retrain the model each time.
So you build a searchable document store.

### Example policy documents

```python
knowledge_docs = [
    "Our return policy allows returns within 30 days of purchase...",
    "Shipping is free for orders over $50...",
    "We offer 24/7 customer support...",
    "All products come with a 1-year warranty...",
]
```

### Split → Embed → Store

```python
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embeddings = FastEmbedEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

### Wrap it as a tool

```python
@tool
def company_policies(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
```

The agent now has an “open book” for policy questions.

---

## **3. Add Middleware (Logging & Observability)**

A Customer Service Agent must:

* Track interactions
* Record tool usage
* Monitor performance
* Store metadata

Your repo includes a clean middleware implementation:

```python
class CustomerServiceMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        print("Processing customer query...")

    def after_model(self, state, runtime):
        print("Response ready")
```

This gives full observability using the official LangChain middleware pattern.

---

## **4. Initialize the Agent**

The heart of the application:

```python
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

system_prompt = """
You are a helpful customer service agent for an e-commerce company.
...
CRITICAL: When using a tool, output ONLY the tool call.
"""

tools = [get_order_status, initiate_return, check_inventory, company_policies]

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[CustomerServiceMiddleware()],
    system_prompt=system_prompt
)
```

This setup ensures the agent:

* Knows its role
* Has access to all tools
* Uses RAG for factual queries
* Provides consistent, professional responses

---

## **5. Handle Customer Queries**

Everything is wrapped in a simple helper with error handling:

```python
def handle_customer(customer_id: str, query: str):
    try:
        # Add recursion_limit to prevent infinite loops
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 10}
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error processing request: {str(e)}"
```

This function:

* Logs events (via middleware)
* Sends the message to the agent
* Handles potential errors gracefully
* Returns the final answer

---

## **6. Run the Live Demo (4 Customer Scenarios)**

Your script runs four complete customer interactions:

### **A. Order status**

```python
"Hi, I'd like to check the status of my order ORD001"
```

The agent correctly calls:

* `get_order_status("ORD001")`

### **B. Return processing**

```python
"I need to return order ORD002..."
```

The agent:

* Retrieves policy text
* Then calls `initiate_return(...)`

### **C. Policy question**

```python
"What's your return policy?"
```

The agent:

* Calls `company_policies()`
* Summarizes the retrieved chunks

### **D. Inventory check**

```python
"Do you have laptops in stock?"
```

The agent:

* Calls `check_inventory("laptop")`

All responses are clean, contextual, and accurate.

---

## **What You Just Built**

You now have a complete, production-style AI system made of:

* **LLM Core**: Groq’s Llama 3.1
* **Tools**: Order, return, inventory functions
* **RAG**: Policy retrieval
* **Logging**: Time, tool usage, metadata
* **Agent Runtime**: LangChain’s latest API
* **End-to-End Workflow**: Multiple customer sessions

This structure is extremely close to what modern AI-powered customer support teams deploy.

You’ve moved from:

A simple “Hello, world” →
to a **full enterprise-grade AI agent.**

---

## **The Journey You Completed**

Here’s a quick recap of all six parts:

### **Part 1:** LLM Basics & Streaming

### **Part 2:** Tools & Agent Actions

### **Part 3:** Memory & Context

### **Part 4:** RAG Pipeline

### **Part 5:** Middleware & Safety

### **Part 6:** Full Customer Service Application

This entire flow mirrors real-world AI agent development.

---

## **What’s Next?**

Now that you’ve mastered the full stack, you can extend this system into:

* A Slack or WhatsApp bot
* A FastAPI backend
* A Django admin tool
* A CLI assistant
* A network automation agent
* A multi-agent coordination system

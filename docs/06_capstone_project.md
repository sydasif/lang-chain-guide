# Part 6: Capstone - The Customer Service Agent

**Goal:** Combine Tools, RAG, Memory, and Logic into a complete Customer Service Bot.

## The Architecture
We are building a bot that can:
1.  **Check Order Status** (Database Tool)
2.  **Process Returns** (Action Tool)
3.  **Answer Policy Questions** (RAG Tool)
4.  **Log Activity** (Middleware)

## The Code Breakdown (`scripts/complete_application.py`)

### The Knowledge Base
We load the shipping policies into a FAISS vector store so the agent can look up rules.

### The Tools
We define three distinct tools:
- `get_order_status(order_id)`
- `initiate_return(order_id)`
- `check_inventory(product)`

### The System Prompt
This is the "Brain". We define the persona and constraints:
```python
system_prompt = """You are a customer service agent.
Responsibilities:
- Help with orders
- Check inventory
- Answer policy questions using the knowledge base
Always be professional."""
```

### The Execution
We run a simulation of a customer session.
1. Customer asks about Order `ORD001`. -> Agent calls `get_order_status`.
2. Customer asks for a return. -> Agent checks policy (RAG) then calls `initiate_return`.

## Conclusion
You have gone from a simple "Hello World" script to a complex, multi-tool agent capable of handling real-world business logic. This is the foundation of modern AI engineering.
# Part 3: Solving the Goldfish Memory Problem

**Goal:** Implement memory so the Agent can have a continuous conversation.

## The Reality of LLMs
LLMs are stateless. If you say "Hi, I'm Bob" and then "What's my name?", the second request knows nothing about the first. We have to send the *entire history* every time.

## Approach 1: Manual History
The simplest way is to manage a list of messages.
From `scripts/agent_with_history.py`:

```python
messages = [
    SystemMessage(content="You are a tutor."),
    HumanMessage(content="My name is Bob."),
    AIMessage(content="Hello Bob!"),
    HumanMessage(content="What is my name?")
]
response = llm.invoke(messages)
```

## Approach 2: Checkpointing (The Pro Way)
In production, you use a "Checkpointer" (like a database) to save the state of a thread.

From `scripts/memory_with_checkpointing.py`:
```python
# Create a config with a thread_id
config = {"configurable": {"thread_id": "session_123"}}

# The agent automatically loads previous messages for this thread_id
result = agent.invoke({"messages": [input_msg]}, config=config)
```

## Context Passing
Sometimes you need to inject invisible data (like User ID or Location) that isn't part of the conversation text but is needed by tools.
From `scripts/context_passing.py`, we see how to pass `user_id` alongside the message payload.

**Next Step:** Now the agent remembers us. But how does it know about our company policies? [Part 4: RAG](./04_rag_pipeline.md).
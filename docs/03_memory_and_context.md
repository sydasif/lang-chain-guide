# LangChain Zero-to-Hero Part 3: Solving the Goldfish Memory Problem

**Welcome to Part 3!** You've built an AI that can converse (Part 1) and take action using tools (Part 2). But there's a critical flaw that makes your agent frustrating to use in real conversations.

Try this experiment with your current agent:

```
You: "Hi, my name is Sarah."
AI: "Hello Sarah! Nice to meet you."

You: "What's my name?"
AI: "I don't have information about your name."
```

**What just happened?** Your AI has the memory of a goldfish. Every interaction is completely isolated—it forgets everything the moment the conversation ends.

This is the single biggest barrier between a demo and a production-ready agent. Let's fix it.

---

## Why LLMs Have Amnesia (The Technical Reality)

Here's the uncomfortable truth: **LLMs are completely stateless**.

When you send a message to an LLM, it's like talking to someone who just woke up with no memory. The model doesn't "remember" previous conversations—it only sees what you send in the current request.

Think of it like this:

### Without Memory

```
Request 1: "My name is Bob"
→ AI processes this standalone
→ Response: "Hello Bob!"
→ [Everything is forgotten]

Request 2: "What's my name?"
→ AI sees only this question
→ Response: "I don't know your name."
```

This isn't a bug—it's fundamental to how LLMs work. They're pure functions: same input always produces the same output, with no hidden state.

**So how do we create the illusion of memory?** We send the entire conversation history with every request.

---

## Approach 1: Manual History Management (The Foundation)

The simplest solution is to maintain a list of messages and send everything each time. Let's see how this works.

Create `agent_with_history.py`:

```python
import getpass
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7, timeout=None, max_retries=2
)

messages = [
    SystemMessage(content="You are a helpful math tutor."),
    HumanMessage(content="What's 25 times 4?"),
    AIMessage(content="25 times 4 equals 100."),
    HumanMessage(content="And what about 100 divided by 5?"),
]

response = llm.invoke(messages)
print(response.content)
```

### Breaking Down the Message Types

LangChain uses three distinct message types:

**1. SystemMessage** - The AI's instructions

```python
SystemMessage(content="You are a helpful math tutor.")
```

This sets the agent's role and behavior. It's like giving someone a job description before they start work.

**2. HumanMessage** - What the user said

```python
HumanMessage(content="What's 25 times 4?")
```

This represents input from the user.

**3. AIMessage** - What the AI responded

```python
AIMessage(content="25 times 4 equals 100.")
```

This captures the AI's previous responses so it can maintain context.

### The Conversation Flow

When you run this script, here's what the AI sees:

```
System: "You are a helpful math tutor."
Human: "What's 25 times 4?"
AI: "25 times 4 equals 100."
Human: "And what about 100 divided by 5?"
```

Notice how the AI can reference the previous calculation (100) because we included it in the message history. This creates continuity.

### Run It

```bash
python agent_with_history.py
```

You'll see something like:

```
100 divided by 5 equals 20.
```

The AI understood that "100" referred to the previous answer because we gave it the full context.

---

## Understanding the Memory Pattern

This manual approach reveals an important principle:

**Memory in AI = Resending relevant history with each request**

Let's visualize the difference:

### Without Memory (Stateless)

```
┌─────────────┐
│  Request 1  │ → AI → Response 1
└─────────────┘

┌─────────────┐
│  Request 2  │ → AI → Response 2  [No connection to Request 1]
└─────────────┘
```

### With Memory (Stateful Simulation)

```
┌─────────────┐
│  Request 1  │ → AI → Response 1
└─────────────┘
        ↓ (stored)
┌─────────────────────────────┐
│ Request 1 + Response 1 +    │ → AI → Response 2
│ Request 2                    │
└─────────────────────────────┘
```

Each new request includes the growing conversation history.

---

## Approach 2: Checkpointing (The Production Solution)

Manual history works for demos, but production applications need something more robust. What happens when:

- The user refreshes the page?
- The conversation spans multiple days?
- You have thousands of concurrent users?

This is where **checkpointing** comes in—saving conversation state to persistent storage (like a database).

Create `memory_with_checkpointing.py`:

```python
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


@tool
def get_user_info(user_id: str) -> str:
    """Pretends to look up user info."""
    if user_id == "user-session-abc123":
        return "User info: Name is Alice, lives in Seattle."
    return "User not found."


tools = [get_user_info]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)


print("User: My name is Alice and I live in Seattle")
messages = [
    {"role": "user", "content": "My name is Alice and I live in Seattle"}
]
result1 = agent.invoke({"messages": messages})
print(f"AI: {result1['messages'][-1].content}")

messages.append(result1['messages'][-1])  # AI response
messages.append({"role": "user", "content": "What's my name and where do I live?"})  # New user query

print("\nUser: What's my name and where do I live?")
result2 = agent.invoke({"messages": messages})
print(f"AI: {result2['messages'][-1].content}")
```

### The Checkpointing Pattern

Notice how we're building the conversation incrementally:

**Turn 1:**

```python
messages = [
    {"role": "user", "content": "My name is Alice and I live in Seattle"}
]
result1 = agent.invoke({"messages": messages})
```

**Turn 2:**

```python
messages.append(result1['messages'][-1])  # Add AI's response
messages.append({"role": "user", "content": "What's my name and where do I live?"})

result2 = agent.invoke({"messages": messages})
```

Each turn, we:

1. Take the previous message history
2. Add the AI's last response
3. Add the user's new message
4. Send everything back to the agent

### Run It

```bash
python memory_with_checkpointing.py
```

You'll see:

```
User: My name is Alice and I live in Seattle
AI: Nice to meet you, Alice! It's great to know you're from Seattle...

User: What's my name and where do I live?
AI: Your name is Alice and you live in Seattle.
```

**Success!** The agent remembered the conversation context.

---

## Real-World Checkpointing Architecture

In production, you'd store this conversation in a database with a thread ID:

```python
# Pseudo-code for production pattern
thread_id = "user_123_session_456"

# Load existing history from database
messages = database.get_conversation(thread_id)

# Add new user message
messages.append({"role": "user", "content": user_input})

# Get AI response
result = agent.invoke({"messages": messages})

# Save updated conversation
database.save_conversation(thread_id, messages + [result['messages'][-1]])
```

This enables:

- **Persistence**: Conversations survive server restarts
- **Multi-session**: Users can pick up where they left off
- **Analytics**: Track conversation patterns and quality
- **Debugging**: Replay conversations to find issues

---

## The Context Window Challenge

There's a catch with infinite memory: **AI models have token limits**.

Most models can handle 4,000-128,000 tokens (~3,000-100,000 words) depending on the model. Long conversations eventually exceed this.

### Strategies to Handle Long Conversations

**1. Sliding Window** (Keep last N messages)

```python
MAX_MESSAGES = 20
messages = messages[-MAX_MESSAGES:]  # Only keep recent context
```

**2. Summarization** (Compress old messages)

```python
if len(messages) > 30:
    summary = summarize_conversation(messages[:20])
    messages = [SystemMessage(summary)] + messages[20:]
```

**3. Semantic Filtering** (Keep only relevant messages)

```python
relevant_messages = filter_by_relevance(messages, current_query)
```

We'll explore advanced memory management in later parts.

---

## Context Passing: The Hidden Data Channel

Sometimes your agent needs information that shouldn't be part of the visible conversation. Examples:

- User ID for personalization
- Geographic location
- Account type or permissions
- Session metadata

This is where **context passing** shines.

Create `context_passing.py`:

```python
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


@tool
def locate_user(user_id: str) -> str:
    """Find a user's location based on their ID."""
    users_db = {
        "user_123": "New York",
        "user_456": "Los Angeles",
        "user_789": "Chicago",
    }
    return users_db.get(user_id, "Unknown location")


@tool
def get_local_weather(location: str) -> str:
    """Get weather for a specific location."""
    return f"Weather in {location}: Partly cloudy, 68°F"


tools = [locate_user, get_local_weather]
agent = create_agent(
    model=llm, tools=tools, system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Locate user user_123 and then get the weather for their location",
        }
    ]
})

print(result["messages"][-1].content)
```

### The Power of Tool Chaining

Notice what happens when you run this:

1. **AI reads the request**: "Locate user_123 and get their weather"
2. **AI calls first tool**: `locate_user("user_123")` → "New York"
3. **AI calls second tool**: `get_local_weather("New York")` → "Partly cloudy, 68°F"
4. **AI synthesizes**: Combines both results into a natural response

The agent automatically chains tools together based on logical dependencies!

### Run It

```bash
python context_passing.py
```

You'll see:

```
User user_123 is located in New York, where the weather is partly cloudy and 68°F.
```

---

## Real-World Context Passing Example

In production, you'd pass invisible metadata:

```python
# User asks: "What's the weather like?"
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather like?"}
    ],
    "context": {
        "user_id": "user_123",
        "location": "New York",
        "timezone": "America/New_York",
        "subscription_tier": "premium"
    }
})
```

Your tools can access this context without exposing it in the conversation:

```python
@tool
def get_weather(context: dict) -> str:
    """Get weather for the user's location."""
    location = context.get("location", "unknown")
    return fetch_weather(location)
```

This enables personalization without cluttering the chat history.

---

## Memory Patterns for Different Use Cases

### Customer Support (Session-Based)

```python
# New conversation each support ticket
thread_id = f"ticket_{ticket_id}"
messages = []  # Fresh start
```

### Personal Assistant (Long-Term)

```python
# Persistent history across days
thread_id = f"user_{user_id}"
messages = database.load_history(thread_id, limit=100)
```

### Collaborative Agents (Shared Context)

```python
# Multiple users, one conversation
thread_id = f"team_{team_id}_discussion"
messages = database.load_shared_history(thread_id)
```

---

## Common Memory Pitfalls (And How to Avoid Them)

### Pitfall 1: Forgetting to Append AI Responses

```python
# WRONG
messages.append({"role": "user", "content": user_input})
result = agent.invoke({"messages": messages})
# Missing: messages.append(result['messages'][-1])

# RIGHT
messages.append({"role": "user", "content": user_input})
result = agent.invoke({"messages": messages})
messages.append(result['messages'][-1])  # Remember to save AI response!
```

### Pitfall 2: Exceeding Token Limits

```python
# Monitor conversation length
total_tokens = estimate_tokens(messages)
if total_tokens > MAX_TOKENS:
    messages = trim_conversation(messages)
```

### Pitfall 3: Leaking Context Between Users

```python
# WRONG - Global messages list shared by all users
messages = []

# RIGHT - User-specific message storage
messages = get_user_messages(user_id)
```

---

## Your Challenge: Build a Personality-Persistent Chatbot

Before Part 4, build this:

**Requirements:**

1. The AI remembers the user's name
2. The AI remembers the user's favorite color
3. After 3 messages, the AI references something mentioned earlier
4. Add a `clear_memory()` function to reset the conversation

**Bonus Points:**

- Store conversation history in a JSON file
- Load previous conversations when restarting the script
- Add a `/history` command that shows the last 5 messages

This exercise will cement your understanding of state management.

---

## Official LangChain Memory Patterns (Production Approach)

The manual history management you've learned is great for understanding how memory works. But LangChain provides official patterns for production applications that go beyond just message history.

### The AgentState Pattern

In production, you often need to track more than just conversation messages. You might need:

- User preferences
- Session metadata
- Custom application state
- Workflow status

LangChain's `AgentState` lets you define custom state that persists across agent invocations.

### Approach 1: Using state_schema

The simplest way to add custom state is with the `state_schema` parameter:

```python
from langchain.agents import AgentState, create_agent
from langchain_groq import ChatGroq

# Define your custom state
class CustomState(AgentState):
    user_preferences: dict
    session_id: str

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Create agent with custom state
agent = create_agent(
    model=llm,
    tools=[],
    state_schema=CustomState
)

# Now you can pass additional state
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}],
    "user_preferences": {"language": "en", "verbosity": "detailed"},
    "session_id": "session_123"
})
```

### What Just Happened?

1. **Extended AgentState**: Created `CustomState` that includes messages (inherited) plus custom fields
2. **Passed Custom State**: Sent `user_preferences` and `session_id` along with messages
3. **State Persists**: The agent can access this state across multiple turns

### Real-World Example: Personalized Assistant

```python
from langchain.agents import AgentState, create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Define state with user context
class UserContextState(AgentState):
    user_name: str
    user_role: str
    preferences: dict

@tool
def get_personalized_greeting(user_name: str, user_role: str) -> str:
    """Generate a personalized greeting based on user context."""
    if user_role == "admin":
        return f"Welcome back, Administrator {user_name}! You have full system access."
    elif user_role == "engineer":
        return f"Hello {user_name}! Ready to configure some networks?"
    else:
        return f"Hi {user_name}!"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

agent = create_agent(
    model=llm,
    tools=[get_personalized_greeting],
    state_schema=UserContextState,
    system_prompt="You are a helpful assistant. Use the user's context to personalize responses."
)

# First interaction
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}],
    "user_name": "Alice",
    "user_role": "engineer",
    "preferences": {"theme": "dark", "notifications": True}
})

print(result["messages"][-1].content)
# Output: Personalized greeting for Alice the engineer
```

### Why Use AgentState?

**1. Type Safety**
Your IDE knows what fields exist and their types.

**2. Validation**
Pydantic validates the state structure automatically.

**3. Documentation**
The state schema serves as documentation for what data the agent needs.

**4. Flexibility**
Easy to add new fields as your application grows.

### Comparison: Manual vs Official Patterns

**Manual History Management (What You Learned):**

```python
# Good for: Understanding, simple apps, full control
messages = []
messages.append({"role": "user", "content": "Hi"})
result = agent.invoke({"messages": messages})
messages.append(result["messages"][-1])
```

**Pros:**

- Simple and explicit
- Full control over what's stored
- Easy to understand

**Cons:**

- Manual state management
- No type safety
- Limited to messages only

**AgentState Pattern (Production):**

```python
# Good for: Production apps, complex state, type safety
class MyState(AgentState):
    user_id: str
    session_data: dict

agent = create_agent(model=llm, tools=tools, state_schema=MyState)
result = agent.invoke({
    "messages": messages,
    "user_id": "user_123",
    "session_data": {...}
})
```

**Pros:**

- Type-safe state management
- Validation built-in
- Scales to complex applications
- Integrates with middleware

**Cons:**

- More setup required
- Slightly more complex for beginners

### When to Use Each Approach

**Use Manual History** when:

- Learning how memory works
- Building simple prototypes
- You only need message history
- You want full control

**Use AgentState** when:

- Building production applications
- You need state beyond messages
- You want type safety and validation
- You're using middleware (Part 5)

---

## What You've Mastered

Incredible progress! You've now learned:

✅ Why LLMs are stateless and how memory works
✅ Manual history management with message types
✅ Production-ready checkpointing patterns
✅ Context passing for hidden metadata
✅ Tool chaining with contextual awareness
✅ Memory management strategies for scale

Your agent now has continuity—it can maintain context across multiple turns.

---

## What's Next: Teaching Your Agent Domain Knowledge

Your agent can remember conversations, but what about knowledge it was never trained on?

**Questions your agent CAN'T answer right now:**

- "What's our company's return policy?"
- "What did the CEO say in last week's meeting?"
- "Where can I find the Q3 financial report?"

The AI doesn't have access to your private documents, company policies, or proprietary information.

**In Part 4**, we'll solve this with **RAG (Retrieval-Augmented Generation)**—teaching your agent to search through your documents and provide accurate, grounded answers.

This is where your agent transforms from a conversational partner into a **knowledge expert**.

---

**Ready to give your agent domain expertise?** Continue to Part 4 where we build a RAG pipeline that lets your AI access and reason over your private documents.

---

*Building something cool with memory? Share your implementation in the comments! I'd love to see creative uses of context passing.*

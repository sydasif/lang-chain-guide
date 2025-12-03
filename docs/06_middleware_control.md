# LangChain Zero-to-Hero Part 6: Building the "Manager Layer" (Middleware & Safety Controls)

**Welcome to Part 6!** You've built an impressive AI agent that can converse ([Part 1](./01_hello_world.md)), take action with tools ([Part 2](./02_tools_and_action.md)), adapt its behavior dynamically ([Part 3](./03_dynamic_behavior.md)), remember conversations ([Part 4](./04_memory_and_context.md)), and access domain knowledge through RAG ([Part 5](./05_rag_pipeline.md)).

But there's a critical problem that will bite you the moment you deploy to production.

## The $10,000 Wake-Up Call

Imagine this scenario:

```bash
Monday 9:00 AM - You deploy your agent to production
Monday 9:15 AM - Users love it! Traffic spikes
Monday 10:30 AM - Your phone buzzes: "API usage alert: $347 in the last hour"
Monday 11:45 AM - Another alert: "$1,240 total spend"
Tuesday 8:00 AM - You wake up to an email: "Your API bill is $10,847"
```

**What happened?** Your agent had no limits. A bug caused it to enter an infinite loop, calling the LLM thousands of times. Or maybe a user asked a complex question that triggered 500 tool calls. Or perhaps someone intentionally abused your system.

This isn't hypothetical—it happens to real developers every week.

## The Missing Layer: Control

Your agent is powerful, but power without control is dangerous. Right now, your agent can:

- **Call expensive APIs unlimited times** → Exploding costs
- **Execute sensitive operations automatically** → Security risks
- **Generate massive outputs** → Performance issues
- **Accumulate huge conversation histories** → Memory problems
- **Run without oversight** → No visibility into what it's doing

Before we build the capstone application in Part 7, we need to add the **manager layer**—middleware that supervises your agent's behavior.

Think of middleware like a responsible manager who:

- **Sets budgets** (rate limiting)
- **Requires approval for big decisions** (human-in-the-loop)
- **Tracks performance** (logging and analytics)
- **Optimizes resources** (summarization)
- **Prevents disasters** (safety controls)

This is what separates hobby projects from production systems.

---

## What Is Middleware? (The Airport Security Analogy)

Middleware sits **between** your user and your agent, intercepting every interaction:

```
User Request
    ↓
[Middleware Layer] ← Checks, logs, controls
    ↓
Agent (Tools + Memory + RAG)
    ↓
[Middleware Layer] ← Inspects, modifies, records
    ↓
Response to User
```

Think of it like airport security:

- **Check bags** → Validate inputs
- **Approve boarding** → Human approval for sensitive actions
- **Flag suspicious items** → Detect anomalies
- **Limit traffic** → Rate limiting
- **Track passengers** → Logging and analytics

Every request passes through security checkpoints before reaching the agent, and every response is inspected before reaching the user.

---

## Your First Middleware: Logging Agent Behavior

Let's start with the most essential middleware—logging. Before you can optimize or control your agent, you need to **see what it's doing**.

### The Problem

Run your agent from Part 2 and ask it a question. What do you see?

```bash
python agent_with_tools.py
```

Output:

```
The weather in London is rainy and 55°F.
```

That's it. You have no idea:

- How long the model took to respond
- How many times it called the model
- What tools it used
- When it started or finished

In production, this blind spot is unacceptable. You need visibility.

### The Solution: LoggingMiddleware

LangChain provides an official `AgentMiddleware` base class that you can extend to intercept agent behavior. Let's build logging middleware.

Create `05.a_logging_middleware.py`:

```python
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Logging Middleware Demo - Official AgentMiddleware")
print("=" * 60)


# Define a simple tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "seattle" in city.lower():
        return "sunny and 75°F"
    if "boston" in city.lower():
        return "cloudy and 68°F"
    return "weather is unknown for that location."


# Create official logging middleware using AgentMiddleware class
class LoggingMiddleware(AgentMiddleware):
    """Official middleware for logging agent behavior."""

    def __init__(self):
        self.call_count = 0
        self.start_times = {}

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Called once when agent starts."""
        print("\n🚀 Agent session started")
        print(f"📝 Initial query: {state['messages'][0].content}")
        return None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Called before each model invocation."""
        self.call_count += 1
        self.start_times[self.call_count] = datetime.now()

        print(f"\n🤖 Model Call #{self.call_count}")
        print(f"📊 Messages in context: {len(state['messages'])}")
        print(
            f"⏰ Started at: {self.start_times[self.call_count].strftime('%H:%M:%S')}"
        )
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called after each model response."""
        end_time = datetime.now()
        duration = (end_time - self.start_times[self.call_count]).total_seconds()

        last_message = state['messages'][-1]
        content_preview = (
            last_message.content[:80] if last_message.content else "No content"
        )

        print(f"✅ Model Call #{self.call_count} completed")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"💬 Response preview: {content_preview}...")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called once when agent completes."""
        print("\n🏁 Agent session completed")
        print(f"📈 Total model calls: {self.call_count}")
        return None


# Create agent with logging middleware
print("\n[Step 1] Creating agent with LoggingMiddleware...")
agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[LoggingMiddleware()],
    system_prompt="You are a helpful weather assistant.",
)
print("✓ Agent created with middleware")

# Test: Simple query
print("\n" + "=" * 60)
print("Test: Weather Query")
print("=" * 60)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Seattle?"}]
})

print(f"\n📤 Final Response:\n{result['messages'][-1].content}")
```

### Breaking Down the Magic

**1. The AgentMiddleware Class**

```python
class LoggingMiddleware(AgentMiddleware):
    """Official middleware for logging agent behavior."""
```

This is the official LangChain pattern. By extending `AgentMiddleware`, you get access to lifecycle hooks.

**2. The Four Lifecycle Hooks**

```python
def before_agent(self, state: AgentState, runtime: Runtime):
    """Runs ONCE when the agent starts."""

def before_model(self, state: AgentState, runtime: Runtime):
    """Runs BEFORE EACH model call."""

def after_model(self, state: AgentState, runtime: Runtime):
    """Runs AFTER EACH model response."""

def after_agent(self, state: AgentState, runtime: Runtime):
    """Runs ONCE when the agent completes."""
```

These hooks automatically intercept the agent's execution at specific points.

**3. Passing Middleware to the Agent**

```python
agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[LoggingMiddleware()],  # ← Add middleware here
    system_prompt="You are a helpful weather assistant.",
)
```

The middleware is now active for every agent invocation!

### Run It

```bash
python 05.a_logging_middleware.py
```

You'll see detailed output like:

```text
============================================================
Logging Middleware Demo - Official AgentMiddleware
============================================================

[Step 1] Creating agent with LoggingMiddleware...
✓ Agent created with middleware

============================================================
Test: Weather Query
============================================================

🚀 Agent session started
📝 Initial query: What's the weather in Seattle?

🤖 Model Call #1
📊 Messages in context: 1
⏰ Started at: 08:15:23

✅ Model Call #1 completed
⏱️  Duration: 0.87s
💬 Response preview: I'll check the weather in Seattle for you...

🤖 Model Call #2
📊 Messages in context: 3
⏰ Started at: 08:15:24

✅ Model Call #2 completed
⏱️  Duration: 0.52s
💬 Response preview: The weather in Seattle is sunny and 75°F...

🏁 Agent session completed
📈 Total model calls: 2

📤 Final Response:
The weather in Seattle is sunny and 75°F.
```

### What Just Happened?

Notice the agent made **2 model calls**:

1. **First call**: The agent analyzed the question and decided to use the `get_weather` tool
2. **Second call**: After getting the tool result, it formatted a natural response

Without middleware, you'd never know this! The logging middleware revealed:

- **Session lifecycle**: When the agent started and finished
- **Performance metrics**: Each call took ~0.5-0.9 seconds
- **Context growth**: Messages increased from 1 to 3 (user query + tool call + tool result)
- **Total cost**: 2 model calls instead of 1

This visibility is **essential** for:

- Debugging slow responses
- Optimizing performance
- Tracking API costs
- Monitoring production systems

---

## Rate Limiting: Protecting Your Wallet

Remember the $10,000 wake-up call from the introduction? Let's prevent that with rate limiting middleware.

### The Problem

Your agent from Part 2 has no limits. Try this thought experiment:

```python
# Hypothetical bug that causes infinite loop
while True:
    agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

Each invocation costs money. At $0.001 per call, 10,000 calls = $10. A million calls = $1,000. This can happen faster than you think.

**Real scenarios that cause cost spikes:**

- Bug in your code creates a loop
- User asks a complex question triggering hundreds of tool calls
- Malicious user intentionally spams your API
- Load testing without limits
- Forgotten background job running continuously

### The Solution: RateLimitMiddleware

Let's build middleware that limits API calls to a safe threshold.

Create `05.b_rate_limiting_middleware.py`:

```python
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Rate Limiting Middleware Demo")
print("=" * 60)


@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    return f"Found results for: {query}"


# Official rate limiting middleware using AgentMiddleware
class RateLimitMiddleware(AgentMiddleware):
    """Middleware to limit the number of model calls per minute."""

    def __init__(self, max_calls_per_minute: int = 5):
        self.max_calls = max_calls_per_minute
        self.calls = []
        print(f"⚙️  Rate limit set to {max_calls_per_minute} calls/minute")

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check rate limit before each model call."""
        now = datetime.now()

        # Remove calls older than 60 seconds
        self.calls = [t for t in self.calls if (now - t).total_seconds() < 60]

        # Check if limit exceeded
        if len(self.calls) >= self.max_calls:
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call).total_seconds()

            print("\n🚫 Rate limit exceeded!")
            print(f"⏳ Please wait {wait_time:.1f} seconds")
            raise Exception(
                f"Rate limit exceeded. "
                f"Maximum {self.max_calls} calls per minute. "
                f"Please wait {wait_time:.1f} seconds."
            )

        # Record this call
        self.calls.append(now)
        remaining = self.max_calls - len(self.calls)
        print(f"\n✅ Rate limit check passed ({remaining} calls remaining)")

        return None


# Create agent with rate limiting
print("\n[Step 1] Creating agent with rate limiting...")
agent = create_agent(
    model=llm,
    tools=[search_database],
    middleware=[RateLimitMiddleware(max_calls_per_minute=3)],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created with rate limit: 3 calls/minute")

# Test: Make multiple calls to trigger rate limit
print("\n" + "=" * 60)
print("Testing Rate Limit (3 calls/minute)")
print("=" * 60)

for i in range(1, 6):
    print(f"\n--- Call {i} ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Search for item {i}"}]
        })
        print(f"✅ Call {i} succeeded")
        print(f"Response: {result['messages'][-1].content[:60]}...")
    except Exception as e:
        print(f"❌ Call {i} failed: {str(e)}")
        break

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
```

### Breaking Down the Logic

**1. Tracking Call Timestamps**

```python
def __init__(self, max_calls_per_minute: int = 5):
    self.max_calls = max_calls_per_minute
    self.calls = []  # List of timestamps
```

The middleware maintains a list of when each call happened.

**2. The Sliding Window Algorithm**

```python
# Remove calls older than 60 seconds
self.calls = [t for t in self.calls if (now - t).total_seconds() < 60]
```

This creates a "sliding window"—only calls in the last 60 seconds count toward the limit.

**3. Enforcing the Limit**

```python
if len(self.calls) >= self.max_calls:
    raise Exception("Rate limit exceeded...")
```

If too many calls happened recently, block the request.

### Run It

```bash
python 05.b_rate_limiting_middleware.py
```

You'll see output like:

```text
============================================================
Rate Limiting Middleware Demo
============================================================
⚙️  Rate limit set to 3 calls/minute

[Step 1] Creating agent with rate limiting...
✓ Agent created with rate limit: 3 calls/minute

============================================================
Testing Rate Limit (3 calls/minute)
============================================================

--- Call 1 ---

✅ Rate limit check passed (2 calls remaining)
✅ Call 1 succeeded
Response: I found results for item 1 in the database...

--- Call 2 ---

✅ Rate limit check passed (1 calls remaining)
✅ Call 2 succeeded
Response: Here are the results for item 2...

--- Call 3 ---

✅ Rate limit check passed (0 calls remaining)
✅ Call 3 succeeded
Response: I found information about item 3...

--- Call 4 ---

🚫 Rate limit exceeded!
⏳ Please wait 58.3 seconds
❌ Call 4 failed: Rate limit exceeded. Maximum 3 calls per minute. Please wait 58.3 seconds.
```

### What Just Happened?

The middleware allowed 3 calls, then blocked the 4th. This prevents:

- **Cost explosions**: Even if your code has a bug, you can't spend more than `(max_calls * cost_per_call * 60 * 24)` per day
- **API abuse**: Malicious users can't overwhelm your system
- **Accidental loops**: Infinite loops are automatically stopped

### Real-World Cost Savings

Let's do the math:

**Without rate limiting:**

- Bug causes 100,000 calls in an hour
- At $0.001 per call = $100/hour
- Over a weekend = $4,800

**With rate limiting (10 calls/minute):**

- Maximum 600 calls/hour
- At $0.001 per call = $0.60/hour
- Over a weekend = $28.80

**Savings: $4,771.20** from one middleware!

---

## Human-in-the-Loop: Requiring Approval for Sensitive Actions

Some actions are too important to happen automatically. Let's add a safety checkpoint.

### The Problem

Imagine your agent from Part 2 with an email tool:

```python
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email."""
    # Actually sends the email!
    return f"Email sent to {recipient}"
```

Now imagine a user asks: *"Email my boss and tell them I quit."*

Without human approval, the agent **will actually send that email**. This is terrifying for:

- Sending emails or messages
- Deleting data or resources
- Making financial transactions
- Modifying network configurations
- Executing system commands
- Approving purchases

You need a human checkpoint before these actions execute.

### The Solution: HumanApprovalMiddleware

Let's build middleware that intercepts sensitive tool calls and asks for approval.

Create `05.c_human_approval_middleware.py`:

```python
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Human-in-the-Loop Middleware Demo")
print("=" * 60)


# Define tools
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email. This is a sensitive operation."""
    return f"Email sent to {recipient} with subject '{subject}'"


@tool
def check_calendar(date: str) -> str:
    """Checks the calendar for a given date."""
    return f"No events scheduled for {date}"


# Official middleware for human approval
class HumanApprovalMiddleware(AgentMiddleware):
    """Middleware to require human approval for sensitive tools."""

    def __init__(self, sensitive_tools: list[str]):
        self.sensitive_tools = sensitive_tools

    def wrap_tool_call(self, tool_call: Any, handler: Callable[[Any], Any]) -> Any:
        """Intercept tool calls and check for sensitive operations."""
        # Extract tool name from tool_call
        try:
            tool_name = getattr(tool_call, 'name', None) or tool_call.get('name')
            tool_args = getattr(tool_call, 'args', None) or tool_call.get('args')
        except Exception:
            tool_name = "unknown"
            tool_args = {}

        # Check if tool is sensitive
        if tool_name in self.sensitive_tools:
            print(f"\n⚠️  SENSITIVE ACTION DETECTED: {tool_name}")
            print(f"   Args: {tool_args}")

            # Request approval
            approval = input("   Do you approve this action? (yes/no): ")

            if approval.lower() != "yes":
                print("   ❌ Action rejected by user")
                return f"Error: User rejected the action '{tool_name}'."

            print("   ✅ Action approved")

        # Proceed with execution
        return handler(tool_call)


# Create agent with approval middleware
print("\n[Step 1] Creating agent with approval middleware...")
agent = create_agent(
    model=llm,
    tools=[send_email, check_calendar],
    middleware=[HumanApprovalMiddleware(sensitive_tools=["send_email"])],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created (send_email requires approval)")

# Test 1: Sensitive tool (should ask for approval)
print("\n" + "=" * 60)
print("Test 1: Sensitive Tool (send_email)")
print("=" * 60)
print("Note: Type 'yes' to approve, 'no' to reject")

result1 = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Send an email to boss@example.com saying 'I quit'",
        }
    ]
})
print(f"\nAI: {result1['messages'][-1].content}")

# Test 2: Safe tool (should run automatically)
print("\n" + "=" * 60)
print("Test 2: Safe Tool (check_calendar)")
print("=" * 60)

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Check my calendar for tomorrow"}]
})
print(f"\nAI: {result2['messages'][-1].content}")
```

### Breaking Down the Pattern

**1. The wrap_tool_call Hook**

```python
def wrap_tool_call(self, tool_call: Any, handler: Callable[[Any], Any]) -> Any:
    """Intercept tool calls BEFORE they execute."""
```

This hook wraps around tool execution, giving you control over whether the tool runs.

**2. Identifying Sensitive Tools**

```python
def __init__(self, sensitive_tools: list[str]):
    self.sensitive_tools = sensitive_tools
```

You specify which tools require approval when creating the middleware.

**3. The Approval Flow**

```python
if tool_name in self.sensitive_tools:
    approval = input("Do you approve this action? (yes/no): ")

    if approval.lower() != "yes":
        return f"Error: User rejected the action '{tool_name}'."
```

The middleware pauses execution and waits for human input.

### Run It

```bash
python 05.c_human_approval_middleware.py
```

**Test 1 Output (Sensitive Tool):**

```text
============================================================
Test 1: Sensitive Tool (send_email)
============================================================
Note: Type 'yes' to approve, 'no' to reject

⚠️  SENSITIVE ACTION DETECTED: send_email
   Args: {'recipient': 'boss@example.com', 'subject': 'I quit', 'body': '...'}
   Do you approve this action? (yes/no): no
   ❌ Action rejected by user

AI: I was unable to send the email because you rejected the action.
```

**Test 2 Output (Safe Tool):**

```text
============================================================
Test 2: Safe Tool (check_calendar)
============================================================

AI: You have no events scheduled for tomorrow.
```

### What Just Happened?

The middleware:

1. **Intercepted** the `send_email` tool call before it executed
2. **Displayed** the tool name and arguments for review
3. **Waited** for human approval
4. **Blocked** the action when you typed "no"
5. **Allowed** the `check_calendar` tool to run automatically (not in sensitive list)

This prevents catastrophic mistakes!

### Real-World Use Cases

**Network Automation:**

```python
sensitive_tools=["apply_config", "reload_device", "delete_vlan"]
```

**DevOps:**

```python
sensitive_tools=["delete_resource", "scale_down", "terminate_instance"]
```

**Finance:**

```python
sensitive_tools=["transfer_funds", "approve_payment", "close_account"]
```

**Customer Service:**

```python
sensitive_tools=["issue_refund", "cancel_subscription", "delete_account"]
```

---

## Summarization Middleware: Managing Context Windows

Long conversations eventually hit a wall—the model's context limit. Let's solve this with automatic summarization.

### The Problem

Remember from Part 3 how conversation history grows with each turn?

```python
messages = [
    {"role": "user", "content": "Hi, my name is Alice"},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "I live in Seattle"},
    {"role": "assistant", "content": "Great!"},
    # ... 50 more exchanges ...
]
```

After a long conversation:

- **Context window fills up**: Models have limits (4K-128K tokens)
- **Costs increase**: More tokens = higher bills
- **Performance degrades**: Longer context = slower responses
- **Eventually fails**: Exceeding the limit causes errors

### The Solution: SummarizationMiddleware

Let's build middleware that automatically compresses conversation history when it gets too long.

Create `05.d_summarization_middleware.py`:

```python
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Summarization Middleware Demo")
print("=" * 60)


@tool
def get_info(topic: str) -> str:
    """Gets information on a topic."""
    return f"Information about {topic}."


# Official summarization middleware
class SummarizationMiddleware(AgentMiddleware):
    """Middleware to summarize history when it exceeds a threshold."""

    def __init__(self, model, max_messages: int = 5):
        self.model = model
        self.max_messages = max_messages

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check history length and summarize if needed."""
        messages = state["messages"]

        if len(messages) > self.max_messages:
            print(f"\n🗜️  History too long ({len(messages)} msgs). Summarizing...")

            # Create summarization prompt
            summary_prompt = (
                "Summarize the following conversation history concisely, "
                "preserving key facts and context:\n\n" + str(messages)
            )

            # Generate summary
            summary = self.model.invoke(summary_prompt).content
            print(f"📝 Summary generated: {summary[:50]}...")

            # Replace history with summary
            new_messages = [
                SystemMessage(content=f"Previous conversation summary: {summary}"),
                messages[-1],  # Keep the last user message
            ]

            # Update state
            state["messages"] = new_messages
            print(f"✅ History compressed to {len(new_messages)} messages")

        return None


# Create agent with summarization
print("\n[Step 1] Creating agent with summarization (max 4 messages)...")
agent = create_agent(
    model=llm,
    tools=[get_info],
    middleware=[SummarizationMiddleware(model=llm, max_messages=4)],
    system_prompt="You are a helpful assistant.",
)
print("✓ Agent created")

# Test: Run a long conversation
print("\n" + "=" * 60)
print("Testing Automatic Summarization")
print("=" * 60)

messages = []

# Simulate a conversation
for i in range(1, 7):
    print(f"\n--- Turn {i} ---")
    user_msg = {"role": "user", "content": f"Tell me about topic {i}"}

    # Add user message to history
    messages.append(user_msg)

    result = agent.invoke({"messages": messages})

    # Update our local history with the state from the agent
    messages = result["messages"]

    response = messages[-1]
    print(f"AI: {response.content}")
```

### Breaking Down the Logic

**1. Monitoring Message Count**

```python
def before_model(self, state: AgentState, runtime: Runtime):
    messages = state["messages"]

    if len(messages) > self.max_messages:
        # Trigger summarization
```

Before each model call, check if history is too long.

**2. Generating the Summary**

```python
summary_prompt = (
    "Summarize the following conversation history concisely, "
    "preserving key facts and context:\n\n" + str(messages)
)

summary = self.model.invoke(summary_prompt).content
```

Use the LLM itself to create a concise summary of the conversation.

**3. Replacing History**

```python
new_messages = [
    SystemMessage(content=f"Previous conversation summary: {summary}"),
    messages[-1],  # Keep the last user message
]

state["messages"] = new_messages
```

Replace all old messages with a single summary message, plus the current user query.

### Run It

```bash
python 05.d_summarization_middleware.py
```

You'll see output like:

```text
============================================================
Summarization Middleware Demo
============================================================

[Step 1] Creating agent with summarization (max 4 messages)...
✓ Agent created

============================================================
Testing Automatic Summarization
============================================================

--- Turn 1 ---
AI: Here's information about topic 1...

--- Turn 2 ---
AI: Topic 2 is interesting because...

--- Turn 3 ---
AI: Let me tell you about topic 3...

--- Turn 4 ---
AI: Topic 4 relates to...

--- Turn 5 ---

🗜️  History too long (5 msgs). Summarizing...
📝 Summary generated: The user asked about topics 1-4. The assistant...
✅ History compressed to 2 messages

AI: Here's what you need to know about topic 5...

--- Turn 6 ---

🗜️  History too long (4 msgs). Summarizing...
📝 Summary generated: Previous topics covered 1-5. The user is now...
✅ History compressed to 2 messages

AI: Topic 6 is fascinating...
```

### What Just Happened?

After turn 4, the conversation had 5+ messages. The middleware:

1. **Detected** the threshold was exceeded
2. **Summarized** all previous messages into a single summary
3. **Replaced** the history with: summary + current user message
4. **Reduced** memory usage from 5 messages to 2

This keeps happening automatically as the conversation grows!

### The Benefits

**Cost Savings:**

- 100-message conversation without summarization: ~50,000 tokens
- With summarization (every 10 messages): ~5,000 tokens
- **Savings: 90% reduction in token costs**

**Performance:**

- Shorter context = faster model responses
- No context limit errors
- Consistent latency regardless of conversation length

**Memory:**

- Prevents out-of-memory errors
- Enables infinite-length conversations
- Maintains relevant context

This is the same strategy used by ChatGPT, Claude, and other production chatbots!

---

## Combining Multiple Middleware

The real power comes from using multiple middleware together:

```python
agent = create_agent(
    model=llm,
    tools=[send_email, search_database, get_weather],
    middleware=[
        LoggingMiddleware(),                          # Track everything
        RateLimitMiddleware(max_calls_per_minute=10), # Prevent cost spikes
        HumanApprovalMiddleware(sensitive_tools=["send_email"]),  # Require approval
        SummarizationMiddleware(model=llm, max_messages=20),      # Manage context
    ],
    system_prompt="You are a helpful assistant.",
)
```

**Execution Order:**

Middleware runs in the order you specify:

1. **before_agent**: Logging → Rate Limit → Approval → Summarization
2. **before_model**: Same order
3. **after_model**: **Reverse order** (Summarization → Approval → Rate Limit → Logging)
4. **after_agent**: Reverse order

This ensures proper cleanup and allows middleware to build on each other.

---

## What You've Mastered

Congratulations! You've just added the critical control layer that separates hobby projects from production systems:

- ✅ **Logging middleware** for complete visibility into agent behavior
- ✅ **Rate limiting** to prevent cost explosions and API abuse
- ✅ **Human-in-the-loop** for sensitive operations requiring approval
- ✅ **Summarization** for managing context windows and reducing costs
- ✅ **Official AgentMiddleware patterns** for production-ready implementations

Your agent now has:

- **Power** (Tools from Part 2)
- **Memory** (Context from Part 3)
- **Knowledge** (RAG from Part 4)
- **Control** (Middleware from Part 6)

---

## Your Challenge: Build a Cost-Tracking Middleware

Before Part 6, try building this custom middleware:

**Requirements:**

1. Track the total number of model calls
2. Estimate token usage (assume 100 tokens per call)
3. Calculate estimated cost (at $0.001 per call)
4. Print a cost report when the agent completes
5. Raise a warning if estimated cost exceeds $1.00

**Hints:**

- Use `before_model` to count calls
- Use `after_agent` to print the final report
- Maintain state in `__init__`

This will reinforce your understanding of middleware lifecycle hooks!

---

## What's Next: The Capstone Project

You now have all the building blocks for a production AI agent:

- **Conversation** (Part 1)
- **Tools** (Part 2)
- **Memory** (Part 3)
- **Knowledge** (Part 4)
- **Control** (Part 6)

**In Part 7**, we combine everything into a **complete, real-world application**:

- Customer service chatbot with RAG knowledge base
- Multiple tools for common operations
- Conversation memory across sessions
- Logging and rate limiting for production
- Human approval for sensitive actions

It's the grand finale where everything comes together!

---

**Ready to build your capstone project?** Continue to [Part 7](./07_capstone_project.md) where we create a production-ready AI application using everything you've learned.

---

*Built something cool with middleware? Share your implementation! Middleware is where AI systems become truly production-ready.*

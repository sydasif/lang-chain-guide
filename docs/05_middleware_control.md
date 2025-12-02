# **LangChain Zero-to-Hero – Part 5: Building the “Manager Layer” (Middleware & Safety Controls)**

Your agent can now talk, remember, use tools, and pull knowledge from documents.
Before we combine everything into a full application, there’s a critical layer we need to add:

> **Control.**

Left unchecked, an agent can:

* Call expensive tools too frequently
* Perform sensitive operations without approval
* Produce long outputs that slow down the app
* Execute actions you didn’t intend
* Run too many API calls and inflate your bill

To prevent this, we introduce **middleware** — the “manager layer” that supervises your agent’s behavior.

This chapter is all about:

* Rate limiting
* Human approval for sensitive actions
* Logging, analytics, and observability
* Post-processing and output inspection
* Response summarization (auto-compression)

Your repository provides practical examples of each.

---

# **What Is Middleware in LangChain?**

Middleware sits between:

* **The user → agent**, and
* **The agent → tools / model**

It allows you to intercept:

* Inputs
* Outputs
* Tool calls
* System messages
* Errors
* Execution flow

Think of it like airport security:

* Check bags
* Approve boarding
* Flag suspicious items
* Limit traffic

This is essential for production-grade AI systems.

---

# **Official LangChain Middleware Patterns**

Before we dive into specific examples, let's understand how LangChain officially implements middleware. This will help you build production-ready systems.

## **The AgentMiddleware Class**

LangChain provides an `AgentMiddleware` base class that you can extend to create custom middleware. This is the **official, production-ready approach**.

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called before each model invocation."""
        print(f"📝 About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called after each model response."""
        print(f"✅ Model responded: {state['messages'][-1].content[:50]}...")
        return None

# Use it with your agent
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[LoggingMiddleware()]
)
```

### **What Just Happened?**

1. **Extended AgentMiddleware**: Created a custom class inheriting from `AgentMiddleware`
2. **Implemented Hooks**: Defined `before_model` and `after_model` methods
3. **Passed to Agent**: Added middleware to the `create_agent` call

The middleware automatically intercepts every model call!

## **Available Hooks**

LangChain provides several hooks you can implement:

**Lifecycle Hooks:**

* `before_agent(state, runtime)` - Runs once when agent starts
* `before_model(state, runtime)` - Runs before each model call
* `after_model(state, runtime)` - Runs after each model response
* `after_agent(state, runtime)` - Runs once when agent completes

**Wrap Hooks:**

* `wrap_model_call(request, handler)` - Wraps entire model call (for retries, fallbacks)
* `wrap_tool_call(request, handler)` - Wraps tool execution (for error handling)

### **When to Use Each Hook**

**Use `before_model` / `after_model` for:**

* Logging each model interaction
* Tracking token usage
* Monitoring performance
* Modifying state between calls

**Use `before_agent` / `after_agent` for:**

* Session initialization
* Final cleanup
* Overall metrics
* One-time setup/teardown

**Use `wrap_model_call` / `wrap_tool_call` for:**

* Retry logic
* Error handling
* Fallback mechanisms
* Circuit breakers

## **Decorator-Based Middleware (Quick Approach)**

For simple middleware, you can use decorators instead of classes:

```python
from langchain.agents.middleware import before_model, after_model

@before_model
def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Calling model with {len(state['messages'])} messages")
    return None

@after_model
def log_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model completed")
    return None

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[log_before, log_after]
)
```

### **Decorator vs Class-Based: When to Use Each**

**Use Decorators When:**

* You need a single hook
* Simple, stateless logic
* Quick prototyping
* No configuration needed

**Use Classes When:**

* Multiple hooks in one middleware
* Need to maintain state
* Require initialization parameters
* Reusing across projects

**Example: Class-based with configuration**

```python
class RateLimitMiddleware(AgentMiddleware):
    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls = max_calls_per_minute
        self.calls = []

    def before_model(self, state, runtime):
        # Check rate limit logic here
        now = datetime.now()
        self.calls = [t for t in self.calls if (now - t).seconds < 60]

        if len(self.calls) >= self.max_calls:
            raise Exception("Rate limit exceeded!")

        self.calls.append(now)
        return None

# Use with custom configuration
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[RateLimitMiddleware(max_calls_per_minute=5)]
)
```

This is much cleaner than manual tracking!

---

# **Practical Middleware Examples**

Now let's see how to implement common middleware patterns using the official API.

---

# **1. Rate Limiting (Protect Your Wallet)**

Your repo includes a clean implementation (`scripts/rate_limiting_middleware.py`).

The logic:

* Track timestamps of recent calls
* Count how many happened in the last 60 seconds
* Block if limit exceeded

```python
class RateLimiter:
    def __init__(self, max_calls_per_minute=5):
        self.max_calls = max_calls_per_minute
        self.calls = []

    def check_rate_limit(self):
        now = datetime.now()
        self.calls = [t for t in self.calls if (now - t).seconds < 60]

        if len(self.calls) >= self.max_calls:
            raise Exception("Rate limit exceeded. Please wait.")

        self.calls.append(now)
```

This is simple but powerful.

In real deployments, rate limiting prevents:

* API spamming
* DDoS-like behavior
* Explosive token usage
* Accidental cost spikes

You enforce it at the app level before hitting the model.

---

# **2. Human-in-the-Loop for Sensitive Operations**

Some actions should not happen automatically:

* Sending emails
* Deleting resources
* Executing commands
* Modifying network devices
* Performing financial transactions

Your example (`scripts/human_in_the_loop_middleware.py`) shows this perfectly:

```python
approval = input("Approve this email? (yes/no): ")
if approval.lower() != "yes":
    raise Exception("Operation cancelled")
```

This turns the tool into a protected action.

Your agent isn’t allowed to act unless a human approves.

This pattern is used in:

* Customer service automation
* Internal IT tools
* DevOps & network automation
* HR bots
* Finance bots

---

# **3. Logging & Observability**

Your sample middleware logs:

* When the model starts
* When the model finishes
* How long it took
* Which tools got called

This gives you visibility into:

* Performance
* Tool usage
* Latency
* Behavior patterns

Example from `scripts/custom_middleware.py`:

```python
print(f"🤖 AI Started at {start_time}")
print(f"📝 Processing prompt: ...")
print(f"🔧 Tool calls and processing completed")
```

Logging becomes crucial when building:

* Production chatbots
* Enterprise tools
* Customer-facing systems

You need to know *exactly* what the agent is doing.

---

# **4. Automated Summarization (Conversation Compression)**

Long chats can grow to hundreds of messages.
This causes:

* Higher costs
* Slower inference
* Memory blow-ups

Your repo includes an elegant solution (`scripts/summarization_middleware.py`).

When the conversation gets too long:

* Take all messages
* Send them to an LLM
* Generate a short summary
* Replace all past messages with the summary

Key logic:

```python
summary = self.model.invoke(summarization_prompt).content
self.messages = [
    SystemMessage(content=f"This is a summary: {summary}")
]
```

Now the conversation stays fresh and manageable.

This is the same strategy used by:

* ChatGPT
* Anthropic Claude
* Replit agents
* Low-latency applications

---

# **Why Middleware Is a Game-Changer**

Everything up to now focused on making the agent powerful.

Middleware focuses on making the agent **safe**.

Without middleware, an agent is like a car without brakes.

With middleware, you gain:

* Control
* Oversight
* Safety
* Stability
* Predictability

This is what separates hobby scripts from production systems.

---

# **Where We Go Next**

At this point, your agent can:

* Talk
* Use tools
* Remember
* Retrieve knowledge
* Follow safety and control logic

The next step is the grand finale.

Part 6 combines everything into a **complete, real-world AI application**:

* Tools
* RAG
* Memory
* Middleware
* Logging
* Customer service workflows
* Multi-step decision logic

It’s the capstone project of the entire series.

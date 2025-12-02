# Part 5: The Manager (Middleware & Safety)

**Goal:** Add guardrails, rate limiting, and human approval to prevent accidents and control costs.

## Middleware Concept
In LangChain, we can wrap our agent logic to intercept inputs and outputs.

## 1. Rate Limiting
To stop a user from spamming your API and draining your wallet (`scripts/rate_limiting_middleware.py`):
```python
def check_rate_limit():
    if len(calls_last_minute) > 5:
        raise Exception("Too many requests! Slow down.")
```

## 2. Human-in-the-Loop
For sensitive actions (like "Send Email"), you don't want the AI to act autonomously. You want a "Confirm?" prompt (`scripts/human_in_the_loop_middleware.py`).

```python
@tool
def send_email(recipient, body):
    """Sends an email."""
    # Pause execution and ask human
    approval = input(f"Approve sending to {recipient}? (yes/no): ")
    if approval != "yes":
        raise Exception("Cancelled")
    return "Email sent"
```

**Next Step:** We have all the pieces. Let's build the final application. [Part 6: Capstone](./06_capstone_project.md).
# LangChain Zero-to-Hero Part 2: Giving Your AI Real Superpowers (Tools & Actions)

**Welcome back!** In [Part 1](./01_hello_world.md), you built your first AI application that could converse and stream responses in real-time. But here's the uncomfortable truth: your AI is currently just a very sophisticated parrot. It can talk, but it can't *do* anything.

Let me show you the problem.

## The Limitation of Pure Conversation

Try asking your current AI:
> "What's the weather in London right now?"

It will confidently give you an answer. But here's the catch—**it's making it up**. The model doesn't have internet access. It doesn't know today's date. It's guessing based on patterns it learned during training.

This is called "hallucination," and it's the Achilles' heel of pure LLM applications.

**So how do we fix this?** We give our AI the ability to use **Tools**.

---

## What Are Tools? (And Why They Change Everything)

A tool is simply a Python function that your AI can call automatically when it needs to perform an action. Think of tools as giving your AI "hands" to interact with the real world.

**Without tools**, your AI is like a person locked in a room with no windows—it can think and talk, but it can't observe or act.

**With tools**, your AI becomes an autonomous agent that can:

- Fetch real-time data from APIs
- Query databases
- Perform calculations
- Execute business logic
- Control external systems

This is the moment your chatbot transforms into an **agent**.

---

## The Anatomy of a Tool

Let's break down what makes a tool work. Here's the simplest possible example:

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "san francisco" in city.lower():
        return "sunny and 72°F"
    if "new york" in city.lower():
        return "cloudy and 65°F"
    if "london" in city.lower():
        return "rainy and 55°F"
    return "weather is unknown for that location."
```

### Three Critical Components

**1. The @tool Decorator**
This special marker tells LangChain: "This function can be called by the AI."

**2. The Docstring (This Is More Important Than You Think!)**

```python
"""Get the current weather for a location."""
```

This isn't just documentation—the AI reads this to decide *when* to use the tool. A vague docstring leads to confused agents. A clear docstring creates intelligent behavior.

**3. Type Hints**

```python
def get_weather(city: str) -> str:
```

These tell the AI what kind of input to provide and what to expect back.

---

## Building Your First Tool-Enabled Agent

Now let's put it all together. Create `agent_with_tools.py`:

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
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "san francisco" in city.lower():
        return "sunny and 72°F"
    if "new york" in city.lower():
        return "cloudy and 65°F"
    if "london" in city.lower():
        return "rainy and 55°F"
    return "weather is unknown for that location."


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in london?"}]
})

print(result["messages"][-1].content)
```

### The Magic Sequence (What Happens Behind the Scenes)

When you run this code, here's the invisible choreography:

1. **User asks**: "What is the weather in London?"
2. **AI analyzes**: It sees a tool called `get_weather` with a docstring about weather
3. **AI decides**: "This question needs that tool!"
4. **LangChain executes**: Runs `get_weather("London")`
5. **Function returns**: "rainy and 55°F"
6. **AI synthesizes**: Crafts a natural response using the real data
7. **User receives**: "The weather in London is currently rainy and 55°F."

This entire process happens automatically. You didn't write any logic to detect weather questions or decide when to call the function—**the AI figured it out**.

---

## Understanding the Agent Flow

Let's visualize what makes this different from Part 1:

### Part 1 (Pure Conversation)

```
User Question → AI → Response
```

### Part 2 (Tool-Enabled Agent)

```
User Question → AI thinks → Calls tool → Gets data → AI synthesizes → Response
```

The AI has gone from reactive to **proactive**. It can now:

- Recognize when it needs external information
- Select the appropriate tool
- Use the tool's output to inform its response

This is the foundation of agent-based AI systems.

---

## Run It and See the Difference

```bash
python agent_with_tools.py
```

You might see output like:

```
The weather in London is currently rainy and 55°F.
```

Notice how the AI:

1. Used the *actual* function result
2. Added natural language around it

Try modifying the user query to:

- "How's the weather in San Francisco?"
- "What should I wear in New York today?"
- "Is it raining in Tokyo?" (This will return "unknown"—a great test!)

---

## Building More Powerful Tools

Our weather function is hardcoded, but in real applications, tools do real work:

### Example: Calculator Tool

```python
@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations. Input should be a valid Python expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

### Example: Database Query Tool

```python
@tool
def get_user_orders(user_id: str) -> str:
    """Retrieve order history for a user from the database."""
    # In reality, you'd query your actual database
    orders = database.query(f"SELECT * FROM orders WHERE user_id = {user_id}")
    return json.dumps(orders)
```

### Example: API Integration Tool

```python
@tool
def search_documentation(query: str) -> str:
    """Search the company documentation for relevant information."""
    response = requests.get(f"https://api.docs.company.com/search?q={query}")
    return response.json()["results"]
```

The pattern is always the same: **Python function + @tool decorator + clear docstring = Agent capability**

---

## Customizing Your Tools (Advanced Patterns)

The basic `@tool` decorator is great for getting started, but LangChain offers powerful customization options for production applications.

### Custom Tool Names

By default, the tool name is the function name. But sometimes you want more control:

```python
@tool("web_search")  # Custom name instead of function name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # Output: web_search
```

**Why custom names?**

- Make tool names more descriptive for the AI
- Avoid naming conflicts
- Use consistent naming conventions across tools

### Custom Tool Descriptions

You can override the docstring with a custom description:

```python
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems."
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

The `description` parameter is what the AI sees when deciding whether to use the tool. Make it clear and specific!

---

## Advanced Schema Definition with Pydantic

For complex tools with multiple parameters, you can use Pydantic models to define precise schemas:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(
    location: str,
    units: str = "celsius",
    include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp}°{units[0].upper()}"

    if include_forecast:
        result += "\nNext 5 days: Sunny"

    return result
```

### Why Use Pydantic Schemas?

**1. Better AI Understanding**
The `Field(description=...)` helps the AI understand exactly what each parameter does.

**2. Type Safety**
`Literal["celsius", "fahrenheit"]` ensures the AI can only pass valid values.

**3. Default Values**
Parameters with defaults are optional—the AI can omit them if not needed.

**4. Validation**
Pydantic automatically validates inputs before your function runs.

### Real-World Example: Network Device Configuration

```python
from pydantic import BaseModel, Field
from typing import Literal

class DeviceConfigInput(BaseModel):
    """Schema for network device configuration."""
    device_ip: str = Field(description="IP address of the network device")
    config_type: Literal["interface", "routing", "acl"] = Field(
        description="Type of configuration to apply"
    )
    config_data: str = Field(description="Configuration commands to apply")
    dry_run: bool = Field(
        default=True,
        description="If True, validate config without applying"
    )

@tool(args_schema=DeviceConfigInput)
def configure_device(
    device_ip: str,
    config_type: str,
    config_data: str,
    dry_run: bool = True
) -> str:
    """Configure a network device with validation."""
    if dry_run:
        return f"[DRY RUN] Would apply {config_type} config to {device_ip}:\n{config_data}"

    # In production, this would use netmiko, NAPALM, or similar
    return f"Applied {config_type} configuration to {device_ip}"
```

This gives the AI:

- Clear parameter descriptions
- Type constraints (only valid config types)
- Safe defaults (dry_run=True prevents accidents)
- Validation before execution

---

## Structured Output: Getting Clean JSON Responses

Sometimes you don't want a paragraph—you want structured data your application can use directly. This is crucial for:

- APIs that consume AI responses
- Mobile apps that display data in specific formats
- Dashboards that need consistent data structures
- Automation pipelines that process AI output

LangChain's structured output feature forces the AI to return data in exactly the format you specify.

### The Power of Pydantic Models

Create `structured_output.py`:

```python
import getpass
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7, timeout=None, max_retries=2
)


# Define your desired output structure using Pydantic
class WeatherResponse(BaseModel):
    location: str = Field(description="The city name")
    temperature: float = Field(description="Temperature in Fahrenheit")
    condition: Literal["sunny", "cloudy", "rainy", "snowy"] = Field(
        description="Weather condition"
    )
    recommendation: str = Field(description="What to wear or do")


# Bind the structure to the model
structured_llm = llm.with_structured_output(WeatherResponse)

# Get structured response
result = structured_llm.invoke("What's the weather in London?")

# Now you have a typed object!
print(f"Location: {result.location}")
print(f"Temperature: {result.temperature}°F")
print(f"Condition: {result.condition}")
print(f"Recommendation: {result.recommendation}")
```

### Breaking Down the Magic

**1. Define Your Schema**

```python
class WeatherResponse(BaseModel):
    location: str = Field(description="The city name")
    temperature: float = Field(description="Temperature in Fahrenheit")
    condition: Literal["sunny", "cloudy", "rainy", "snowy"]
    recommendation: str = Field(description="What to wear or do")
```

This Pydantic model tells the AI: "Your response MUST have these exact fields with these exact types."

**2. Bind the Schema**

```python
structured_llm = llm.with_structured_output(WeatherResponse)
```

This creates a new version of your model that's constrained to your format.

**3. Get Typed Results**

```python
result = structured_llm.invoke("What's the weather in London?")
```

Now `result` is a Python object with actual properties you can access:

```python
Location: London
Temperature: 55.0
Condition: rainy
Recommendation: Wear a waterproof jacket and bring an umbrella.
```

### Why This Matters

Imagine building:

- **A weather dashboard**: You need consistent JSON to populate widgets
- **A mobile app**: You need predictable data structures for your UI
- **An automation pipeline**: You need reliable fields to trigger actions
- **An API endpoint**: You need schema-compliant responses

Structured output eliminates the parsing nightmare and gives you production-ready data.

---

## Real-World Use Cases

Now that you understand tools and structured output, here's what becomes possible:

### Customer Support Agent

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search company documentation for answers to customer questions."""

@tool
def create_support_ticket(issue: str, priority: str) -> str:
    """Create a support ticket in the system."""

@tool
def check_order_status(order_id: str) -> str:
    """Get the current status of a customer order."""
```

### Network Automation Agent

```python
@tool
def check_device_status(ip_address: str) -> str:
    """Check if a network device is online."""

@tool
def get_interface_stats(device: str, interface: str) -> str:
    """Retrieve statistics for a network interface."""

@tool
def apply_configuration(device: str, config: str) -> str:
    """Apply configuration changes to a device."""
```

### Data Analysis Agent

```python
@tool
def query_database(sql: str) -> str:
    """Execute a SQL query and return results."""

@tool
def generate_visualization(data: str, chart_type: str) -> str:
    """Create a chart from data."""

@tool
def export_report(format: str) -> str:
    """Export analysis as PDF or Excel."""
```

---

## The Agent Decision-Making Process

You might be wondering: "How does the AI know which tool to use?"

The answer is surprisingly elegant:

1. **Tool Discovery**: The AI reads all available tool docstrings
2. **Context Analysis**: It analyzes the user's question
3. **Reasoning**: It uses its language understanding to match intent with capability
4. **Execution**: It calls the most relevant tool(s)
5. **Synthesis**: It crafts a natural response using the tool output

This happens in milliseconds, and it's entirely automatic.

---

## Common Patterns and Best Practices

### Pattern 1: Tool Chaining

An agent can call multiple tools in sequence:

```
User: "What's the weather in London and should I visit today?"
→ AI calls get_weather("London")
→ AI calls get_tourist_attractions("London")
→ AI synthesizes both results into a recommendation
```

### Pattern 2: Conditional Logic

```python
@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price. Only use for valid stock symbols."""
    if not is_valid_symbol(symbol):
        return "Invalid stock symbol"
    return fetch_price(symbol)
```

### Pattern 3: Error Handling

```python
@tool
def api_call(endpoint: str) -> str:
    """Make an API call to external service."""
    try:
        response = requests.get(endpoint, timeout=5)
        return response.text
    except Exception as e:
        return f"API call failed: {str(e)}"
```

---

## Debugging Tips

When your agent doesn't behave as expected:

**1. Check Your Docstrings**
Vague: `"""Gets weather."""`
Better: `"""Get the current weather conditions for a specific city."""`

**2. Add Logging**

```python
@tool
def my_tool(input: str) -> str:
    """Tool description."""
    print(f"Tool called with: {input}")  # Debug line
    result = do_something(input)
    print(f"Tool returning: {result}")  # Debug line
    return result
```

**3. Test Tools Independently**

```python
# Test the function directly first
result = get_weather("London")
print(result)  # Should return expected format
```

---

## Your Challenge: Build a Multi-Tool Agent

Before moving to Part 3, try building an agent with THREE tools:

1. **get_current_time()**: Returns the current time
2. **calculate(expression)**: Performs math
3. **flip_coin()**: Returns "heads" or "tails"

Then ask it questions like:

- "What time is it and what's 42 times 7?"
- "Flip a coin three times and tell me the results"
- "If it's after 6pm and the coin is heads, recommend pizza for dinner"

This will force your agent to use multiple tools and combine their results—a critical skill for complex automations.

---

## What You've Mastered

Congratulations! You've just unlocked the core capability that separates chatbots from agents:

- ✅ Created tools from Python functions
- ✅ Built an agent that autonomously decides when to use tools
- ✅ Implemented structured output for production-ready data
- ✅ Understood the agent reasoning loop
- ✅ Learned real-world patterns for tool-based systems

Your AI can now **DO** things, not just talk about them.

---

## What's Next: Memory and Context

But there's still a critical problem: **your agent has amnesia**.

After every conversation, it forgets everything. Ask it "What did I just ask you?" and it has no idea.

**In Part 3**, we'll give your agent a memory by implementing:

- Conversation history tracking
- Persistent memory across sessions
- Thread-based context management
- User-specific personalization

Your agent is about to get a brain that remembers.

---

**Ready to give your agent a memory?** Continue to [Part 3](./03_memory_and_context.md) where we solve the amnesia problem and build truly intelligent, context-aware agents.

---

*Got questions about tools or want to share what you built? Drop a comment below! I love seeing creative tool implementations.*

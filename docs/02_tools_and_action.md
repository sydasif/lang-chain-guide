# Part 2: Giving Your AI "Hands" (Tools)

**Goal:** Transform a chatbot into an Agent that can execute code and return structured data.

## The Problem
LLMs are trapped in a text box. If you ask them "What's the weather?", they hallucinate because they don't have internet access. We solve this with **Tools**.

## Defining a Tool
In LangChain, a tool is just a Python function with a specific decorator (`@tool`) and a clear docstring. The docstring is crucial—it tells the AI *when* to use the tool.

From `scripts/agent_with_tools.py`:
```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    if "london" in city.lower():
        return "rainy and 55°F"
    return "unknown"

# Bind tools to the model
agent = create_agent(model=llm, tools=[get_weather], ...)
```

## Structured Output (JSON)
Sometimes you don't want a conversation; you want raw data for your API. We can force the LLM to output Pydantic objects.

From `scripts/structured_output.py`:
```python
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    condition: str

# The Magic Line
structured_llm = llm.with_structured_output(WeatherResponse)
result = structured_llm.invoke("Weather in London")

print(result.temperature) # Returns a float, not a string!
```

**Next Step:** The agent can use tools, but it forgets who you are immediately. Let's fix that in [Part 3: Memory](./03_memory_and_context.md).
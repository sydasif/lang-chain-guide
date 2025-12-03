"""
Advanced Tool Schema Example - LangChain Zero-to-Hero Part 2

This script demonstrates advanced tool definition using Pydantic schemas
with Field descriptions, Literal types, and default values.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 60)
print("Advanced Tool Schema Demo")
print("=" * 60)


# Define advanced schema with Pydantic
class WeatherInput(BaseModel):
    """Input schema for weather queries."""

    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="Temperature unit preference"
    )
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")


@tool(args_schema=WeatherInput)
def get_weather(
    location: str, units: str = "celsius", include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    # Simulated weather data
    temp_c = 22
    temp_f = 72

    temp = temp_c if units == "celsius" else temp_f
    result = f"Current weather in {location}: {temp}°{units[0].upper()}"

    if include_forecast:
        result += "\n\nNext 5 days forecast:"
        result += "\n- Day 1: Sunny, 24°C"
        result += "\n- Day 2: Partly cloudy, 23°C"
        result += "\n- Day 3: Rainy, 19°C"
        result += "\n- Day 4: Sunny, 25°C"
        result += "\n- Day 5: Sunny, 26°C"

    return result


# Create agent with the advanced tool
print("\n[Step 1] Creating agent with advanced weather tool...")
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant. Use the weather tool to provide accurate information.",
)
print("✓ Agent created")

# Test 1: Simple query (uses defaults)
print("\n" + "=" * 60)
print("Test 1: Simple Weather Query")
print("=" * 60)
print("\nUser: What's the weather in London?")

result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in London?"}]
})
print(f"\nAI: {result1['messages'][-1].content}")

# Test 2: Query with units specified
print("\n" + "=" * 60)
print("Test 2: Weather with Fahrenheit")
print("=" * 60)
print("\nUser: What's the weather in New York in Fahrenheit?")

result2 = agent.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in New York in Fahrenheit?"}
    ]
})
print(f"\nAI: {result2['messages'][-1].content}")

# Test 3: Query with forecast
print("\n" + "=" * 60)
print("Test 3: Weather with Forecast")
print("=" * 60)
print("\nUser: What's the weather in Tokyo? Include the 5-day forecast.")

result3 = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What's the weather in Tokyo? Include the 5-day forecast.",
        }
    ]
})
print(f"\nAI: {result3['messages'][-1].content}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("- Pydantic Field() provides clear descriptions to the AI")
print("- Literal types constrain valid values")
print("- Default values make parameters optional")
print("- The AI intelligently uses the schema to call the tool correctly")

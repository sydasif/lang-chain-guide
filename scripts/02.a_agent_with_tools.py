from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


# Define a custom tool that the agent can use to get weather information
# The @tool decorator makes this function available to the agent
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a location."""
    # Return weather information for specific cities
    # This is a mock implementation for demonstration purposes
    if "san francisco" in city.lower():
        return "sunny and 72°F"
    if "new york" in city.lower():
        return "cloudy and 65°F"
    if "london" in city.lower():
        return "rainy and 55°F"
    # Default response for cities not specified above
    return "weather is unknown for that location."


# Create an agent with the specified model and tools
# llm: The language model to use (ChatGroq with Llama 3.3 70B)
# tools: A list of functions the agent can use (in this case, get_weather)
# system_prompt: Sets the context and role for the agent (weather assistant)
agent = create_agent(
    model=llm, tools=[get_weather], system_prompt="You are a helpful weather assistant."
)

# Invoke the agent with a user query about the weather in London
# The agent will use the get_weather tool to fetch information if needed
result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in london?"}]
})

# Print the final response from the agent
# result["messages"][-1] gets the last message in the conversation
# .content extracts the text content from that message
print(result["messages"][-1].content)

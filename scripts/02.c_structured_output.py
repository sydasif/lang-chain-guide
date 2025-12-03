import getpass
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Prompt user for API key if not found in environment variables
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Initialize the ChatGroq language model
llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0.7, timeout=None, max_retries=2
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

import getpass
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Prompt user for API key if not found in environment variables
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Initialize the ChatGroq language model with specific parameters
# model: Specifies which LLM to use (Llama 3.3 70B versatile model)
# temperature: Controls randomness in output (0.7 provides a balance between creativity and coherence)
# timeout: No timeout specified, allowing requests to take as long as needed
# max_retries: Maximum number of retry attempts if a request fails (set to 2)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7, timeout=None, max_retries=2
)

# Create a list of messages that represent the conversation history
# This includes system instructions, human inputs, and AI responses
# SystemMessage: Provides context and role for the AI (math tutor in this case)
# HumanMessage: Represents questions or inputs from the user
# AIMessage: Represents previous responses from the AI
messages = [
    SystemMessage(content="You are a helpful math tutor."),
    HumanMessage(content="What's 25 times 4?"),
    AIMessage(content="25 times 4 equals 100."),
    HumanMessage(content="And what about 100 divided by 5?"),
]

# Send the entire message history to the language model
# The model uses this context to provide a relevant response to the latest query
response = llm.invoke(messages)
print(response.content)

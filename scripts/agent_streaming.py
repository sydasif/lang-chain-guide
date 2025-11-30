import getpass
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

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

# Print a label for the AI response, without adding a newline
print("AI: ", end="", flush=True)

# Stream the response from the language model in chunks
# This allows for real-time output as the model generates the response
# Each chunk contains a portion of the response content
for chunk in llm.stream("Why is the sky blue?"):
    # Print each chunk immediately without adding newlines
    # flush=True ensures the output appears immediately in the terminal
    print(chunk.content, end="", flush=True)

# Print a final newline after the streaming is complete
print()

# LangChain Complete Guide: Building AI Agents from Scratch

## **A Comprehensive Tutorial for Beginners**

Are you interested in building powerful AI applications but overwhelmed by the complexity of working with different AI models and APIs? LangChain is here to simplify your journey. In this comprehensive guide, we'll walk through everything you need to know to start building AI agents, from basic concepts to advanced features.

## What is LangChain?

LangChain is a Python framework that provides a unified interface for working with various AI models and tools. Think of it as a universal translator that lets you switch between different AI providers (OpenAI, Anthropic, Google) without rewriting your code.

**Why use LangChain?**

- **Standard Model Interface**: LangChain provides a standardized interface for interacting with a wide variety of models, so you can easily swap between them without changing your code.
- **Easy-to-use, Flexible Agents**: LangChain's agent abstraction is designed to be easy to get started with, but also provides the flexibility to customize your agent's behavior.
- **Built on LangGraph**: LangChain's agents are built on top of LangGraph, a low-level agent orchestration framework that provides durable execution, streaming, human-in-the-loop support, and persistence.
- **Debug with LangSmith**: LangSmith provides deep visibility into your agent's behavior, with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.

## LangChain and LangGraph

LangChain is the easiest way to start building agents and applications powered by LLMs. It provides a pre-built agent architecture and model integrations to help you get started quickly.

LangGraph is a low-level agent orchestration framework and runtime that allows you to build more advanced agents with a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.

LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.

---

## Getting Started: Environment Setup

Let's set up your development environment. You'll need Python 3.10 or higher installed on your system.

### Step 1: Install LangChain

```bash
pip install -U langchain
```

**What we're installing:**

- `langchain`: The core framework
- `langchain-openai`: Provider package for OpenAI models
- `python-dotenv`: To manage API keys securely

### Step 2: Set Up API Keys

Create a file named `.env` in your project directory:

```text
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Important**: Never commit your `.env` file to version control. Add it to your `.gitignore` file!

### Step 3: Load Your Environment

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv("GROQ_API_KEY")
```

**Why this matters**: Storing API keys in a `.env` file keeps them separate from your code, making your application more secure and easier to deploy.

---

## Building Your First Simple Agent

Now for the exciting part – let's build an AI agent that can use tools! We'll create a weather assistant that can check weather conditions using the latest LangChain features.

### The Complete Example

```python
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# Step 1: Define a custom tool
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
    model="groq:llama-3.3-70b-versatile",  # Or any other compatible model
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in london?"}]
})

print(result["messages"][-1].content)
```

### How This Works

1. **Tool Definition**: The `@tool` decorator remains the same, transforming a Python function into a tool the agent can use. The docstring is crucial as it tells the AI what the tool does.
2. **Agent Creation**: The new `create_agent` function simplifies the process immensely. You pass the model, a list of tools, and a system prompt directly to the function. LangChain handles the prompt templating and agent setup behind the scenes.
3. **Agent Invocation**: The agent is now invoked with a dictionary containing a `messages` list. This standardized format makes it easier to manage conversation history.

**Expected Output:**

```text
The weather in San Francisco is sunny and 72°F.
```

---

## Working with Standalone Models

Sometimes you don't need a full agent – you just want to chat with an AI model directly. LangChain makes this simple too.

### Basic Model Interaction

```python
import getpass
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7, timeout=None, max_retries=2
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

### Maintaining Conversation History

```python
import getpass
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7, timeout=None, max_retries=2
)

messages = [
    SystemMessage(content="You are a helpful math tutor."),
    HumanMessage(content="What's 25 times 4?"),
    AIMessage(content="25 times 4 equals 100."),
    HumanMessage(content="And what about 100 divided by 5?"),
]

response = llm.invoke(messages)
print(response.content)
```

**Key Insight**: The AI doesn't inherently remember past conversations. You must pass the full message history each time!

### Streaming Responses

For a better user experience, you can stream responses word-by-word:

```python
import getpass
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7, timeout=None, max_retries=2
)

print("AI: ", end="", flush=True)
for chunk in llm.stream("Why is the sky blue?"):
    print(chunk.content, end="", flush=True)
```

**Output:**

```text
AI: Lines of code flow free
Bugs hide in logic's shadow
Debug brings the light
```

---

## Advanced Agent Concepts

Let's enhance our agents with three powerful features: structured output, context passing, and memory.

### 1. Structured Output

Force the model to return data in a specific format using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import Literal

# Define your desired output structure
class WeatherResponse(BaseModel):
    location: str = Field(description="The city name")
    temperature: float = Field(description="Temperature in Fahrenheit")
    condition: Literal["sunny", "cloudy", "rainy", "snowy"] = Field(description="Weather condition")
    recommendation: str = Field(description="What to wear or do")

# Bind the structure to the model
structured_llm = llm.with_structured_output(WeatherResponse)

# Get structured response
result = structured_llm.invoke("What's the weather in London?")

# Now you have a typed object!
print(f"Temperature: {result.temperature}°F")
print(f"Condition: {result.condition}")
print(f"Recommendation: {result.recommendation}")
```

**Why this is powerful**: You get guaranteed data structure, making it easy to integrate with databases, APIs, or UI components.

### 2. Context Passing

Pass additional information to your agent that tools can use:

```python
from langchain.tools import tool

# Tool that uses context
@tool
def locate_user(user_id: str) -> str:
    """Find a user's location based on their ID."""
    # In production, this would query a database
    users_db = {
        "user_123": "New York",
        "user_456": "Los Angeles",
        "user_789": "Chicago"
    }
    return users_db.get(user_id, "Unknown location")

@tool
def get_local_weather(location: str) -> str:
    """Get weather for a specific location."""
    return f"Weather in {location}: Partly cloudy, 68°F"

# Create agent with multiple tools
tools = [locate_user, get_local_weather]
agent = create_agent(
    model="gpt-4",
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

# Invoke with context
result = agent_executor.invoke({
    "input": "What's the weather where I am?",
    "user_id": "user_123"  # Context passed to agent
})

print(result["output"])
```

**How it works**: The agent can call `locate_user` with the `user_id` from context, then call `get_local_weather` with that location!

### 3. Memory with Checkpointing

Give your agent memory across conversations:

```python
from langchain_core.checkpointers import InMemorySaver

# Create a checkpointer to store conversation history
checkpointer = InMemorySaver()

# Create agent executor with memory
agent = create_agent(
    model="gpt-4",
    tools=tools,
    system_prompt="You are a helpful assistant.",
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    checkpointer=checkpointer,
    verbose=True
)

# Configuration to identify the conversation thread
config = {"configurable": {"thread_id": "user-session-abc123"}}

# First message
result1 = agent_executor.invoke(
    {"input": "My name is Alice and I live in Seattle"},
    config=config
)

# Second message - the agent remembers!
result2 = agent_executor.invoke(
    {"input": "What's my name and where do I live?"},
    config=config
)

print(result2["output"])  # "Your name is Alice and you live in Seattle."
```

**The Magic**: The `thread_id` links all messages in the same conversation. Different users get different threads!

---

## Working with Images: Multimodal Input

Modern AI models can process both text and images. Here's how to use them with LangChain.

### Using Image URLs

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Use a vision-capable model
vision_llm = ChatOpenAI(model="gpt-4-vision-preview")

# Create a message with both text and image
message = HumanMessage(
    content=[
        {"type": "text", "text": "What objects do you see in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"}
        }
    ]
)

response = vision_llm.invoke([message])
print(response.content)
```

### Using Local Images (Base64 Encoding)

```python
import base64

def encode_image(image_path: str) -> str:
    """Read an image file and convert it to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode your local image
base64_image = encode_image("path/to/your/image.jpg")

# Create message with base64 image
message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image in detail"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
)

response = vision_llm.invoke([message])
print(response.content)
```

**Use Case Example**: Build an app that analyzes product photos, identifies plants, or helps visually impaired users understand images.

---

## Retrieval Augmented Generation (RAG)

RAG is a technique that gives your AI access to external knowledge. Instead of relying only on training data, the AI can search through your documents to find relevant information.

### The Complete RAG Pipeline

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader

# Step 1: Load your documents
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval with generation.",
    "Vector databases store embeddings, which are numerical representations of text.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Embeddings capture semantic meaning, so similar texts have similar embeddings."
]

# Step 2: Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=200,  # Maximum characters per chunk
    chunk_overlap=20  # Overlap between chunks to maintain context
)
docs = text_splitter.create_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # Return top 2 most relevant documents
)

# Step 5: Convert retriever to a tool for the agent
retriever_tool = create_retriever_tool(
    retriever,
    "knowledge_base",
    "Search for information about LangChain, RAG, and vector databases. "
    "Use this tool whenever you need to answer questions about these topics."
)

# Step 6: Create agent with retriever tool
tools = [retriever_tool]
agent = create_agent(
    model="gpt-4",
    tools=tools,
    system_prompt="You are a helpful assistant.",
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 7: Ask questions that require retrieved knowledge
result = agent_executor.invoke({
    "input": "Explain what RAG is and how it works"
})

print(result["output"])
```

### How RAG Works: A Step-by-Step Breakdown

1. **Document Loading**: Your knowledge base is loaded (could be PDFs, websites, databases)

2. **Chunking**: Documents are split into smaller pieces that fit in the AI's context window

3. **Embedding**: Each chunk is converted to a vector (a list of numbers) that represents its meaning

4. **Storage**: Vectors are stored in a vector database (FAISS in our example)

5. **Query Time**: When a user asks a question:
   - The question is converted to a vector
   - The vector database finds the most similar document chunks
   - These chunks are sent to the AI along with the question
   - The AI generates an answer using the retrieved information

**Real-World Example**: Imagine building a customer support chatbot for your company. You can feed it all your documentation, and it will retrieve relevant sections to answer customer questions accurately.

### Testing the Retriever

```python
# Test direct retrieval
query = "What is FAISS?"
relevant_docs = retriever.get_relevant_documents(query)

print("Retrieved documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n{i}. {doc.page_content}")
```

---

## Middleware: Supercharging Your Agents

Middleware sits between your user's request and the AI's response, allowing you to modify behavior dynamically. Think of it as a layer of smart logic that adapts your agent to different situations.

### 1. Dynamic Prompt Selection

Customize system prompts based on user roles:

```python
from langchain_core.prompts import ChatPromptTemplate

def create_dynamic_agent(user_role: str):
    """Create an agent with role-specific instructions."""

    # Define different prompts for different roles
    prompts = {
        "admin": """You are an administrative assistant with elevated privileges.
                   You can access sensitive information and perform system operations.
                   Be professional and security-conscious.""",

        "customer": """You are a friendly customer service representative.
                      Help users with their questions and guide them to resources.
                      Always be polite and patient.""",

        "developer": """You are a technical assistant for developers.
                       Provide detailed code examples and technical explanations.
                       Use programming terminology appropriately."""
    }

    system_prompt = prompts.get(user_role, prompts["customer"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_agent(
        model="gpt-4",
        tools=tools,
        system_prompt=system_prompt,
    )
    return AgentExecutor(agent=agent, tools=tools)

# Usage
admin_agent = create_dynamic_agent("admin")
customer_agent = create_dynamic_agent("customer")

# Same question, different responses based on role
result = admin_agent.invoke({
    "input": "How do I reset the system?"
})
```

### 2. Dynamic Model Selection

Switch between models based on conversation complexity:

```python
def get_smart_model(message_count: int, conversation_history: list):
    """Select appropriate model based on conversation state."""

    # For short conversations, use faster, cheaper model
    if message_count <= 5:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # For longer conversations, use more capable model
    elif message_count <= 15:
        return ChatOpenAI(model="gpt-4", temperature=0.7)

    # For very long conversations, use the most advanced model
    else:
        return ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

# Track conversation
conversation_history = []
message_count = 0

def chat(user_input: str):
    global message_count, conversation_history

    message_count += 1
    conversation_history.append({"role": "user", "content": user_input})

    # Select model dynamically
    current_model = get_smart_model(message_count, conversation_history)

    # Create agent with selected model
    agent = create_agent(
        model=current_model,
        tools=tools,
        system_prompt="You are a helpful assistant.",
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    result = agent_executor.invoke({"input": user_input})

    conversation_history.append({"role": "assistant", "content": result["output"]})
    return result["output"]

# Usage
print(chat("Hello!"))  # Uses GPT-3.5
print(chat("Tell me more"))  # Still GPT-3.5
# ... after 5 messages ...
print(chat("Complex question"))  # Automatically switches to GPT-4!
```

### 3. Custom Middleware with Callbacks

Create sophisticated logging and monitoring:

```python
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime
import json

class AdvancedMiddleware(BaseCallbackHandler):
    """Custom middleware for logging, analytics, and control."""

    def __init__(self):
        self.call_log = []
        self.start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts processing."""
        self.start_time = datetime.now()
        print(f"\n🤖 AI Started at {self.start_time.strftime('%H:%M:%S')}")
        print(f"📝 Processing prompt: {prompts[0][:100]}...")

    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        print(f"✅ AI Completed in {duration:.2f} seconds")
        print(f"📊 Generated {len(response.generations[0])} response(s)")

        # Log for analytics
        self.call_log.append({
            "timestamp": end_time.isoformat(),
            "duration": duration,
            "prompt_length": len(str(response)),
        })

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool is about to be used."""
        tool_name = serialized.get("name", "Unknown")
        print(f"🔧 Using tool: {tool_name}")
        print(f"   Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes."""
        print(f"   ✓ Tool output: {str(output)[:100]}")

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        print(f"🎯 Agent Action: {action.tool}")
        print(f"   Reasoning: {action.log[:100]}...")

    def get_analytics(self):
        """Get analytics from all calls."""
        if not self.call_log:
            return "No calls recorded yet"

        total_calls = len(self.call_log)
        avg_duration = sum(call["duration"] for call in self.call_log) / total_calls

        return f"""
Analytics Summary:
- Total Calls: {total_calls}
- Average Duration: {avg_duration:.2f} seconds
- Total Time: {sum(call["duration"] for call in self.call_log):.2f} seconds
        """

# Usage
middleware = AdvancedMiddleware()

agent = create_agent(
    model="gpt-4",
    tools=tools,
    system_prompt="You are a helpful assistant.",
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[middleware],
    verbose=True
)

# Make some calls
agent_executor.invoke({"input": "What's the weather in Seattle?"})
agent_executor.invoke({"input": "And in Boston?"})

# View analytics
print(middleware.get_analytics())
```

### 4. Other Powerful Middleware Patterns

#### Rate Limiting

```python
class RateLimitMiddleware(BaseCallbackHandler):
    def __init__(self, max_calls_per_minute=10):
        self.max_calls = max_calls_per_minute
        self.calls = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if (now - t).seconds < 60]

        if len(self.calls) >= self.max_calls:
            raise Exception("Rate limit exceeded. Please wait.")

        self.calls.append(now)
```

#### Human-in-the-Loop

```python
class HumanApprovalMiddleware(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")

        # Require approval for sensitive operations
        if tool_name in ["delete_data", "send_email", "make_payment"]:
            approval = input(f"Approve {tool_name} with input '{input_str}'? (yes/no): ")
            if approval.lower() != "yes":
                raise Exception("Operation cancelled by user")
```

#### Automatic Conversation Summarization

```python
class SummarizationMiddleware(BaseCallbackHandler):
    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.message_count = 0

    def on_llm_end(self, response, **kwargs):
        self.message_count += 1

        if self.message_count >= self.max_messages:
            # Trigger summarization
            print("🗜️ Compressing conversation history...")
            # Implementation would summarize and reset conversation
            self.message_count = 0
```

---

## Putting It All Together: A Complete Application

Let's build a comprehensive customer service agent that uses everything we've learned:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.checkpointers import InMemorySaver
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime

# 1. Define Tools
@tool
def get_order_status(order_id: str) -> str:
    """Get the status of a customer order."""
    orders = {
        "ORD001": "Shipped - arriving tomorrow",
        "ORD002": "Processing - will ship today",
        "ORD003": "Delivered on Nov 28"
    }
    return orders.get(order_id, "Order not found")

@tool
def initiate_return(order_id: str, reason: str) -> str:
    """Start a return process for an order."""
    return f"Return initiated for order {order_id}. Reason: {reason}. Return label sent to email."

@tool
def check_inventory(product_name: str) -> str:
    """Check if a product is in stock."""
    inventory = {
        "laptop": "In stock - 15 units available",
        "phone": "Low stock - 3 units remaining",
        "tablet": "Out of stock - restocking Dec 5"
    }
    return inventory.get(product_name.lower(), "Product not found")

# 2. Create Knowledge Base (RAG)
knowledge_docs = [
    "Our return policy allows returns within 30 days of purchase for a full refund.",
    "Shipping is free for orders over $50. Standard shipping takes 3-5 business days.",
    "We offer 24/7 customer support via chat, email, and phone.",
    "All products come with a 1-year manufacturer warranty.",
    "You can track your order using the order number sent to your email."
]

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(knowledge_docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

retriever_tool = create_retriever_tool(
    retriever,
    "company_policies",
    "Search company policies, shipping info, and customer service guidelines."
)

# 3. Set Up Middleware
class CustomerServiceMiddleware(BaseCallbackHandler):
    def __init__(self):
        self.interactions = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"\n{'='*50}")
        print(f"🕐 {datetime.now().strftime('%H:%M:%S')} - Processing customer query...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        print(f"🔧 Accessing: {tool_name}")

    def on_llm_end(self, response, **kwargs):
        print(f"✅ Response ready")
        self.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "response_length": len(str(response))
        })

# 4. Initialize Agent
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer service agent for an e-commerce company.

    Your responsibilities:
    - Help customers with order inquiries
    - Process returns and exchanges
    - Check product inventory
    - Answer policy questions using the knowledge base

    Be friendly, professional, and efficient. Always verify order IDs before processing requests."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [get_order_status, initiate_return, check_inventory, retriever_tool]
agent = create_agent(
    model="gpt-4",
    tools=tools,
    system_prompt="""You are a helpful customer service agent for an e-commerce company.

    Your responsibilities:
    - Help customers with order inquiries
    - Process returns and exchanges
    - Check product inventory
    - Answer policy questions using the knowledge base

    Be friendly, professional, and efficient. Always verify order IDs before processing requests.""",
)

checkpointer = InMemorySaver()
middleware = CustomerServiceMiddleware()

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    checkpointer=checkpointer,
    callbacks=[middleware],
    verbose=True
)

# 5. Run Customer Service Sessions
def handle_customer(customer_id: str, query: str):
    """Handle a customer service interaction."""
    config = {"configurable": {"thread_id": f"customer-{customer_id}"}}

    result = agent_executor.invoke(
        {"input": query},
        config=config
    )

    return result["output"]

# Example Interactions
print("\n" + "="*70)
print("CUSTOMER SERVICE AGENT - LIVE DEMO")
print("="*70)

# Session 1: Order Status
response1 = handle_customer(
    "CUST123",
    "Hi, I'd like to check the status of my order ORD001"
)
print(f"\n📞 Agent: {response1}")

# Session 2: Return Request
response2 = handle_customer(
    "CUST456",
    "I need to return order ORD002, the product doesn't fit"
)
print(f"\n📞 Agent: {response2}")

# Session 3: Policy Question
response3 = handle_customer(
    "CUST789",
    "What's your return policy?"
)
print(f"\n📞 Agent: {response3}")

# Session 4: Inventory Check
response4 = handle_customer(
    "CUST321",
    "Do you have laptops in stock?"
)
print(f"\n📞 Agent: {response4}")

print("\n" + "="*70)
print("SESSION COMPLETE")
print("="*70)
```

### What This Application Demonstrates

1. **Multiple Tools**: Order status, returns, inventory checks, and knowledge base search
2. **RAG Integration**: Answers policy questions using retrieved company documentation
3. **Memory**: Maintains conversation context per customer using checkpointer
4. **Middleware**: Logs all interactions for analytics and monitoring
5. **Professional Prompting**: Clear role definition and behavioral guidelines

---

## Best Practices and Tips

### 1. Tool Design

- **Clear Docstrings**: The AI reads your docstrings to understand tools
- **Type Hints**: Use proper type annotations for parameters
- **Error Handling**: Tools should handle errors gracefully

```python
@tool
def search_database(query: str, limit: int = 5) -> str:
    """
    Search the product database.

    Args:
        query: Search term (product name, category, or SKU)
        limit: Maximum number of results to return (default: 5)

    Returns:
        JSON string with search results
    """
    try:
        # Your database logic here
        results = perform_search(query, limit)
        return json.dumps(results)
    except Exception as e:
        return f"Search failed: {str(e)}"
```

### 2. Prompt Engineering

- **Be Specific**: Clearly define the agent's role and capabilities
- **Set Boundaries**: Tell the agent what it should NOT do
- **Provide Examples**: Few-shot examples improve accuracy

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical support agent for a software company.

    Your role:
    - Diagnose and resolve technical issues.
    - Escalate complex problems to the engineering team.
    - Provide clear, step-by-step instructions.

    What you must NOT do:
    - Do not offer discounts or refunds.
    - Do not discuss future product roadmaps.
    - Do not ask for personally identifiable information (PII) like passwords.

    Example interaction:
    User: "My app is crashing on startup."
    Agent: "I'm sorry to hear that. Let's try clearing the app cache. Here are the steps..."
    """),
    ("human", "{input}")
])
```

### 3. Security

- **Validate Inputs**: Never trust user input directly. Sanitize and validate data before passing it to tools.
- **Limit Permissions**: Give agents and tools the minimum permissions they need to function.
- **Monitor Activity**: Use tools like LangSmith to track agent behavior and detect anomalies.

---

## Middleware

Middleware allows you to add custom logic to your agent's execution. You can use it to:

- **Log requests and responses**: Keep track of your agent's activity for debugging and analysis.
- **Implement caching**: Store the results of expensive operations to improve performance.
- **Add authentication and authorization**: Control who can access your agent and what they can do.
- **And much more**: The possibilities are endless!

## Human-in-the-loop

LangChain's agents are built on top of LangGraph, which provides support for human-in-the-loop workflows. This means you can pause your agent's execution and wait for human input before continuing. This is useful for:

- **Getting feedback on your agent's performance**: Ask users to rate your agent's responses and use their feedback to improve your agent's performance.
- **Asking for clarification**: If your agent is unsure how to proceed, you can ask the user for more information.
- **And more**: The possibilities are endless!

## Multi-agent systems

LangChain's agents can be combined to create multi-agent systems. This allows you to build more complex agents that can work together to solve problems. For example, you could create a team of agents that includes a researcher, a writer, and an editor. The researcher could be responsible for finding information, the writer could be responsible for writing a draft, and the editor could be responsible for reviewing and editing the draft.

## Retrieval

LangChain's agents can be combined with retrieval-augmented generation (RAG) to create agents that can answer questions about your data. This is useful for:

- **Building a customer support bot**: Your bot can answer questions about your products and services.
- **Building a research assistant**: Your assistant can help you find information about any topic.
- **And more**: The possibilities are endless!

## Long-term memory

LangChain's agents can be combined with long-term memory to create agents that can remember information from previous conversations. This is useful for:

- **Building a personal assistant**: Your assistant can remember your preferences and provide personalized recommendations.
- **Building a customer support bot**: Your bot can remember your customers' previous interactions and provide more personalized support.
- **And more**: The possibilities are endless!

## Deployment and Production

Taking your LangChain application from a local prototype to a production environment requires careful planning. Here are key considerations:

### 1. Hosting and Infrastructure

- **Serverless Functions (AWS Lambda, Google Cloud Functions)**: Ideal for event-driven or low-traffic applications. Cost-effective and auto-scaling.
- **Containers (Docker, Kubernetes)**: Best for complex applications with multiple services. Provides consistency across environments.
- **Virtual Private Servers (VPS)**: Offers more control but requires manual setup and maintenance.

### 2. Cost Management

- **Model Selection**: Use cheaper models (like GPT-3.5) for simple tasks and more expensive models (like GPT-4) for complex reasoning.
- **Caching**: Cache responses for common queries to reduce redundant API calls.
- **Usage Monitoring**: Set up alerts to notify you of unexpected spikes in API usage.

### 3. Security in Production

- **API Key Management**: Use a secure secret management service (like AWS Secrets Manager or HashiCorp Vault) instead of `.env` files.
- **Authentication and Authorization**: Protect your application endpoints to ensure only authorized users can access them.
- **Input/Output Guardrails**: Implement strict validation to prevent prompt injection and other security vulnerabilities.

### 4. Deploy with LangSmith

LangSmith is essential for production-grade AI applications. It helps you:

- **Trace Execution**: Visualize the entire lifecycle of your agent's requests.
- **Evaluate Performance**: Test your agents against predefined datasets to measure accuracy.
- **Monitor Errors**: Get alerts on failures and debug issues quickly.
- **Gather Feedback**: Allow users to provide feedback on AI responses to improve your system over time.

### 5. Observability

LangSmith provides deep visibility into your agent's behavior, with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.

---

## Conclusion

LangChain is a powerful framework that dramatically simplifies the development of AI-powered applications. In this guide, we've covered the entire journey from setting up your environment to building a complete, production-ready customer service agent.

**Key Takeaways:**

- **Modularity is Key**: LangChain's components (models, tools, prompts) can be mixed and matched to create custom solutions.
- **Agents are Decision-Makers**: They use AI to reason and select the right tools for a given task.
- **RAG Provides Knowledge**: Retrieval Augmented Generation gives your AI access to external, up-to-date information.
- **Middleware Adds Control**: Customize your agent's behavior with dynamic logic for prompts, models, and security.
- **Production Requires Planning**: Deploying AI is more than just code; it involves infrastructure, security, and monitoring.

The world of AI is evolving rapidly, and LangChain is at the forefront of this change. By mastering these concepts, you are well-equipped to build the next generation of intelligent applications.

---

## Available Scripts in This Repository

This repository includes several example scripts that demonstrate various LangChain concepts. All scripts have been updated to work with the current LangChain API:

### Basic Agent Scripts

- **`agent_with_tools.py`**: A simple weather assistant that demonstrates tool usage with the latest LangChain agent API.
- **`agent_streaming.py`**: Shows how to stream responses from LLMs for better user experience.
- **`agent_with_history.py`**: Demonstrates maintaining conversation history with AI models.

### Advanced Feature Scripts

- **`context_passing.py`**: Illustrates how to pass additional context to agents for enhanced functionality.
- **`custom_middleware.py`**: Shows implementation of custom logging and analytics middleware.
- **`dynamic_model_selection.py`**: Demonstrates switching between different models based on conversation complexity.
- **`dynamic_prompt_selection.py`**: Shows how to dynamically select system prompts based on user roles.

### Memory and RAG Scripts

- **`memory_with_checkpointing.py`**: Implements conversation memory across multiple interactions.
- **`rag_pipeline.py`**: Complete Retrieval Augmented Generation pipeline with vector database integration.

### Middleware Examples

- **`human_in_the_loop_middleware.py`**: Implements human approval workflows for sensitive operations.
- **`rate_limiting_middleware.py`**: Shows how to implement rate limiting to control API usage.

### Complete Application

- **`complete_application.py`**: A comprehensive customer service agent combining multiple tools, RAG, memory, and middleware.

Each script is self-contained and can be run independently. They showcase best practices for implementing LangChain agents with the current API.

## Next Steps

Your journey with LangChain is just beginning! Here are some resources to continue your learning:

- **Official Documentation**: The [LangChain Python Docs](https://python.langchain.com/) are the single best source for in-depth information.
- **LangChain Cookbook**: Explore a collection of [community-contributed examples](https://github.com/langchain-ai/langchain/tree/master/cookbook) for various use cases.
- **LangChain Blog**: Stay updated with the latest features and tutorials on the [official blog](https://blog.langchain.dev/).
- **Community Discord**: Join the [LangChain Discord](https://discord.gg/langchain) to ask questions and connect with other developers.

Happy building!

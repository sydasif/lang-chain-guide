import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")


# Define a simple tool
@tool
def get_info(topic: str) -> str:
    """Gets information on a topic."""
    return f"Information about {topic}."


tools = [get_info]


class SummarizationMiddleware(BaseCallbackHandler):
    def __init__(self, model, max_messages=5):
        self.model = model
        self.max_messages = max_messages
        self.messages = []

    def on_chain_end(self, outputs, **kwargs):
        if "messages" in outputs:
            self.messages = outputs["messages"]
            if len(self.messages) >= self.max_messages:
                print(
                    f"\n🗜️ Compressing conversation history ({len(self.messages)} messages)..."
                )

                # Create the summarization prompt
                summarization_prompt = [
                    SystemMessage(content="Summarize the following conversation."),
                    HumanMessage(content=str(self.messages)),
                ]

                # Get the summary
                summary = self.model.invoke(summarization_prompt).content

                # Reset the conversation with the summary
                self.messages = [
                    SystemMessage(
                        content=f"This is a summary of the previous conversation: {summary}"
                    )
                ]
                outputs["messages"] = self.messages
                print(f"   New message history length: {len(self.messages)}")


# Usage
llm = ChatGroq(model="llama-3.3-70b-versatile")
middleware = SummarizationMiddleware(model=llm, max_messages=5)

agent = create_agent(
    model=llm, tools=tools, system_prompt="You are a helpful assistant."
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, callbacks=[middleware], verbose=True
)

# Run a conversation that will trigger summarization
for i in range(6):
    agent_executor.invoke({
        "messages": middleware.messages
        + [HumanMessage(content=f"Tell me about topic {i + 1}")]
    })

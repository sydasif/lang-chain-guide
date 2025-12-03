# LangChain Zero-to-Hero: Educational Repository

## Project Specification

This repository serves as the codebase for the "LangChain Zero-to-Hero" blog series. It is designed to demonstrate the progression from basic LLM inference to a fully functional, production-grade AI Agent with memory, tools, and RAG capabilities.

**Stack:**

- Python 3.12+
- LangChain 0.3+ (Latest API)
- Groq (Llama 3.3) for inference
- FAISS for vector storage
- Pydantic for structured data

## Blog Series (Table of Contents)

This codebase is documented in detail in the following 7-part series located in the `docs/` folder:

1. **[Part 1: The Hello World of AI Agents](./docs/01_hello_world.md)**
    - *Topics:* Environment setup, basic inference, and response streaming.
2. **[Part 2: Giving Your AI "Hands" (Tools)](./docs/02_tools_and_action.md)**
    - *Topics:* The `@tool` decorator, function calling, and structured JSON output.
3. **[Part 3: Making Agents Smarter (Dynamic Behavior)](./docs/03_dynamic_behavior.md)**
    - *Topics:* Dynamic prompts, model selection, conditional tools, and query routing.
4. **[Part 4: Solving the Goldfish Memory Problem](./docs/04_memory_and_context.md)**
    - *Topics:* Manual history, checkpointing, and context passing.
5. **[Part 5: The Open-Book Exam (RAG)](./docs/05_rag_pipeline.md)**
    - *Topics:* Vector stores (FAISS), embeddings, and retrieval-augmented generation.
6. **[Part 6: The Manager (Middleware & Safety)](./docs/06_middleware_control.md)**
    - *Topics:* Rate limiting, human-in-the-loop, and observability.
7. **[Part 7: Capstone - The Customer Service Agent](./docs/07_capstone_project.md)**
    - *Topics:* Building a complete application combining all previous concepts.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` with `GROQ_API_KEY`.
3. Run any script in `scripts/`: `python scripts/complete_application.py`

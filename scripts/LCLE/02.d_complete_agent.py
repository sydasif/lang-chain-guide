#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 8: Complete Production Network Agent
Full implementation with classification, routing, safety, logging, and error handling
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

print("=" * 80)
print("  _   _ _____ _______        _____  _____  _  __")
print(" | \\ | |  ___|__   __|      / ___ \\|  __ \\| |/ /")
print(" |  \\| | |__    | |  ____  | |   | | |__) | ' / ")
print(" | . ` |  __|   | | |____| | |   | |  ___/|  <  ")
print(" | |\\  | |___   | |        | |___| | |    | . \\ ")
print(" |_| \\_|_____|  |_|         \\_____/|_|    |_|\\_\\")
print()
print("Network Automation Agent v1.0 - Production Ready")
print("=" * 80)


# ===== LOGGING SYSTEM =====
class AgentLogger:
    """Comprehensive logging system for audit trail and debugging."""

    def __init__(self):
        self.log_file = f"agent_logs_{datetime.now().strftime('%Y%m%d')}.txt"

    def log_event(self, event_type: str, details: dict):
        """Log an event with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Console output
        print(f"\n[{timestamp}] {event_type}")
        for key, value in details.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            print(f"  {key}: {value_str}")

        # File output
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {event_type}\n")
            for key, value in details.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")


logger = AgentLogger()


# ===== CLASSIFIER =====
def classify_query(text: str) -> str:
    """Classify user intent with logging."""
    network_keywords = [
        "show",
        "display",
        "check",
        "get",
        "interface",
        "ip",
        "bgp",
        "ospf",
        "eigrp",
        "device",
        "switch",
        "router",
        "config",
        "status",
        "vlan",
        "route",
        "arp",
        "mac",
        "ping",
    ]

    text_lower = text.lower()
    matched_keywords = [kw for kw in network_keywords if kw in text_lower]
    is_network = len(matched_keywords) > 0

    category = "network_task" if is_network else "general"

    logger.log_event(
        "CLASSIFICATION",
        {
            "query": text[:100],
            "category": category,
            "matched_keywords": ", ".join(matched_keywords)
            if matched_keywords
            else "none",
        },
    )

    return category


# ===== SAFETY SYSTEM =====
def is_safe_command(command: str) -> tuple[bool, str]:
    """
    Validate command safety.

    Returns:
        (is_safe: bool, reason: str)
    """
    unsafe_terms = [
        "reload",
        "reboot",
        "erase",
        "shutdown",
        "conf t",
        "configure",
        "no ",
        "delete",
        "write erase",
    ]

    command_lower = command.lower()

    # Check for unsafe terms
    for term in unsafe_terms:
        if term in command_lower:
            return False, f"Contains unsafe term: '{term}'"

    # Must start with show
    if not command_lower.startswith("show"):
        return False, "Only 'show' commands allowed"

    return True, "Command is safe"


# ===== NETWORK TOOLS =====
@tool
def run_network_command(device: str, command: str) -> str:
    """
    Execute a network show command on a device.

    Args:
        device: Device hostname or IP (e.g., "switch-core-01")
        command: Show command to execute (e.g., "show ip interface brief")

    Returns:
        Command output or error message
    """

    # Validate safety
    is_safe, reason = is_safe_command(command)

    if not is_safe:
        logger.log_event(
            "COMMAND_BLOCKED", {"device": device, "command": command, "reason": reason}
        )
        return f"❌ BLOCKED: {reason}"

    # Log execution
    logger.log_event(
        "COMMAND_EXECUTION",
        {"device": device, "command": command, "status": "executing"},
    )

    # Simulate command execution (replace with Netmiko/NAPALM in production)
    simulated_output = f"""
{command} on {device}
========================================
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     192.168.1.1     YES NVRAM  up                    up
GigabitEthernet0/1     10.0.0.1        YES NVRAM  up                    up
GigabitEthernet0/2     10.0.1.1        YES NVRAM  down                  down
Loopback0              192.168.100.1   YES NVRAM  up                    up
========================================
"""

    logger.log_event(
        "COMMAND_SUCCESS",
        {"device": device, "command": command, "output_length": len(simulated_output)},
    )

    return simulated_output.strip()


@tool
def ping_device(device: str) -> str:
    """
    Check if a device is reachable.

    Args:
        device: Device hostname or IP address

    Returns:
        Ping result
    """
    logger.log_event("PING_CHECK", {"device": device})

    # Simulate ping (replace with actual ping in production)
    result = f"✅ Device {device} is reachable\n   RTT: min=1ms, avg=3ms, max=5ms\n   Packet loss: 0%"

    logger.log_event("PING_SUCCESS", {"device": device, "result": result})
    return result


# ===== CHAINS =====

# General chat chain
general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant for network engineers.
    When the question is not related to network operations, answer it professionally.
    Keep responses concise and clear.""",
    ),
    ("human", "{question}"),
])

general_chain = general_prompt | llm | (lambda x: x.content)

# Network operations chain
network_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a network automation assistant.
    You have access to tools for checking device status.
    Use the run_network_command tool when users ask about device status.
    Only use 'show' commands - never configuration commands.
    Be clear and professional in your responses.""",
    ),
    ("human", "{question}"),
])

llm_with_tools = llm.bind_tools([run_network_command, ping_device])
network_chain = network_prompt | llm_with_tools


# ===== ROUTING SYSTEM =====

classifier = RunnableLambda(lambda q: {"question": q, "category": classify_query(q)})


def route_query(input_data):
    """Route query to appropriate chain."""
    category = input_data["category"]
    question = input_data["question"]

    logger.log_event(
        "ROUTING",
        {
            "category": category,
            "destination": "network_chain"
            if category == "network_task"
            else "general_chain",
        },
    )

    if category == "network_task":
        return network_chain.invoke({"question": question})
    return general_chain.invoke({"question": question})


router = RunnableLambda(route_query)

# Complete agent pipeline
agent = classifier | router


# ===== INTERACTIVE LOOP =====
def run_agent():
    """Run the agent in interactive mode."""
    print("\n🤖 Network Automation Agent Ready!")
    print("=" * 80)
    print("\n📚 Commands:")
    print("  - Type your question or network command")
    print("  - Type 'help' for examples")
    print("  - Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("\n👋 Goodbye! All actions have been logged to:", logger.log_file)
                break

            if user_input.lower() == "help":
                print("\n" + "=" * 80)
                print("📚 Example Commands")
                print("=" * 80)
                print("\n🔧 Network Commands:")
                print("  - Show interface status on switch-core-01")
                print("  - Check device reachability for router-edge-02")
                print("  - Display ip interface brief on switch-access-05")
                print("  - Get interface statistics from router-core-01")
                print("\n💬 General Questions:")
                print("  - What is OSPF?")
                print("  - Explain BGP routing")
                print("  - Tell me about network automation")
                print("=" * 80)
                continue

            # Process query
            logger.log_event("USER_QUERY", {"query": user_input})

            print("\n🤖 Agent:", end=" ", flush=True)
            result = agent.invoke(user_input)

            # Display result
            if hasattr(result, "content"):
                print(result.content)
                logger.log_event(
                    "AGENT_RESPONSE", {"type": "content", "response": result.content}
                )
            elif hasattr(result, "tool_calls") and result.tool_calls:
                print(f"\n🔧 Executing {len(result.tool_calls)} tool(s)...")
                for tc in result.tool_calls:
                    print(f"   - Tool: {tc['name']}")
                    print(f"   - Arguments: {tc['args']}")
                logger.log_event(
                    "AGENT_RESPONSE",
                    {
                        "type": "tool_calls",
                        "num_tools": len(result.tool_calls),
                        "tools": [tc["name"] for tc in result.tool_calls],
                    },
                )
            else:
                print(result)
                logger.log_event(
                    "AGENT_RESPONSE", {"type": "other", "response": str(result)}
                )

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! All actions have been logged to:", logger.log_file)
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.log_event("ERROR", {"error": str(e), "type": type(e).__name__})


# ===== MAIN =====
if __name__ == "__main__":
    # Run automated tests
    print("\n" + "=" * 80)
    print("RUNNING AUTOMATED TESTS")
    print("=" * 80)

    test_cases = [
        ("Show ip interface brief on switch-core-01", "network_task"),
        ("What's the weather today?", "general"),
        ("Check BGP neighbors on router-edge-02", "network_task"),
        ("Explain OSPF routing protocol", "general"),
        ("Ping device 192.168.1.1", "network_task"),
    ]

    passed = 0
    failed = 0

    for query, expected_category in test_cases:
        print(f"\n{'·' * 80}")
        print(f"📝 Test Query: {query}")
        actual_category = classify_query(query)

        if actual_category == expected_category:
            print(f"✅ PASS (Expected: {expected_category}, Got: {actual_category})")
            passed += 1
        else:
            print(f"❌ FAIL (Expected: {expected_category}, Got: {actual_category})")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)

    # Run interactive mode
    print("\n" + "=" * 80)
    print("STARTING INTERACTIVE MODE")
    print("=" * 80)

    run_agent()

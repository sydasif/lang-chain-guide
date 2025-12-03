#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 8: Query Classifier
Determines if a query needs network tools or can be answered with general knowledge
"""


def classify_query(text: str) -> str:
    """
    Classify user queries into categories.

    Args:
        text: User's question or command

    Returns:
        "network_task" or "general"
    """
    # Keywords that indicate network operations
    network_keywords = [
        # Command verbs
        "show",
        "display",
        "check",
        "get",
        "ping",
        "trace",
        "traceroute",
        # Network components
        "interface",
        "ip",
        "bgp",
        "ospf",
        "eigrp",
        "device",
        "switch",
        "router",
        "config",
        "configuration",
        "status",
        "vlan",
        "route",
        "routing",
        "arp",
        "mac",
        "running",
        "startup",
        "version",
        "neighbor",
        "trunk",
        "access",
    ]

    text_lower = text.lower()

    # Check for network keywords
    for keyword in network_keywords:
        if keyword in text_lower:
            return "network_task"

    # Default to general chat
    return "general"


def classify_with_confidence(text: str) -> tuple[str, float, list[str]]:
    """
    Classify query with confidence score and matched keywords.

    Args:
        text: User's question or command

    Returns:
        (category, confidence, matched_keywords)
    """
    network_keywords = [
        "show",
        "display",
        "check",
        "interface",
        "ip",
        "bgp",
        "ospf",
        "device",
        "switch",
        "router",
        "config",
        "status",
        "vlan",
        "route",
    ]

    text_lower = text.lower()
    matched = [kw for kw in network_keywords if kw in text_lower]

    if matched:
        # More matches = higher confidence
        confidence = min(1.0, len(matched) * 0.3)
        return "network_task", confidence, matched
    return "general", 1.0, []


# ===== TESTS =====
if __name__ == "__main__":
    print("=" * 60)
    print("Query Classifier Test Suite")
    print("=" * 60)

    test_cases = [
        ("Show me the interface status on switch-01", "network_task"),
        ("What's the weather today?", "general"),
        ("Check BGP neighbors on router-core", "network_task"),
        ("Tell me a joke about networking", "general"),
        ("Display running config", "network_task"),
        ("How do I cook pasta?", "general"),
        ("Get ip route table from firewall-01", "network_task"),
        ("What is OSPF?", "general"),
        ("Ping device 192.168.1.1", "network_task"),
    ]

    print("\n[Test 1] Basic Classification")
    print("-" * 60)

    passed = 0
    failed = 0

    for query, expected in test_cases:
        result = classify_query(query)
        status = "✅" if result == expected else "❌"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} Query: {query}")
        print(f"   Expected: {expected}, Got: {result}")

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    print("\n" + "=" * 60)
    print("[Test 2] Classification with Confidence")
    print("-" * 60)

    confidence_test_queries = [
        "Show ip interface brief on switch-core-01",
        "Hello, how are you?",
        "Check device status and show config on router-edge-02",
    ]

    for query in confidence_test_queries:
        category, confidence, keywords = classify_with_confidence(query)
        print(f"\n📝 Query: {query}")
        print(f"   Category: {category}")
        print(f"   Confidence: {confidence:.2f}")
        if keywords:
            print(f"   Matched keywords: {', '.join(keywords)}")

    print("\n" + "=" * 60)
    print("Classifier Test Complete")
    print("=" * 60)

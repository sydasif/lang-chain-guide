#!/usr/bin/env python3
"""
LangChain Zero-to-Hero Part 8: Network Tools
Safe tools for network device interaction with comprehensive safety checks
"""

from typing import Optional

from langchain.tools import tool

# ===== SAFETY CONFIGURATION =====

# Commands that should NEVER be executed automatically
UNSAFE_COMMANDS = [
    "reload",
    "reboot",
    "restart",
    "erase",
    "write erase",
    "conf t",
    "configure terminal",
    "configure",
    "no ",
    "shutdown",
    "delete",
    "format",
    "clear",
]


def is_safe_command(command: str) -> tuple[bool, str]:
    """
    Validate that a command is safe to execute.

    Args:
        command: The command to validate

    Returns:
        (is_safe: bool, reason: str)
    """
    command_lower = command.lower()

    # Check for unsafe terms
    for unsafe_term in UNSAFE_COMMANDS:
        if unsafe_term in command_lower:
            return False, f"Contains unsafe term: '{unsafe_term}'"

    # Only allow show commands (read-only operations)
    if not command_lower.startswith("show"):
        return False, "Only 'show' commands are permitted for safety"

    # Additional validation: check command length
    if len(command) > 200:
        return False, "Command too long (max 200 characters)"

    return True, "Command is safe to execute"


# ===== NETWORK TOOLS =====


@tool
def run_network_command(device: str, command: str) -> str:
    """
    Execute a network command on a device.

    Use this tool when the user wants to check device status, view configurations,
    or retrieve information from network equipment.

    IMPORTANT: This tool only accepts READ-ONLY commands (show commands).
    Configuration changes are not permitted.

    Args:
        device: Device hostname or IP address (e.g., "switch-core-01", "192.168.1.1")
        command: Show command to execute (e.g., "show ip interface brief")

    Returns:
        Command output or error message
    """

    # Input validation
    if not device or len(device) < 3:
        return "❌ Error: Invalid device name provided"

    if not command:
        return "❌ Error: No command provided"

    # Safety check
    is_safe, reason = is_safe_command(command)

    if not is_safe:
        return f"❌ BLOCKED: {reason}\n\nOnly 'show' commands are allowed for safety."

    # Log the operation
    print("\n🔧 [TOOL EXECUTION]")
    print(f"   Device: {device}")
    print(f"   Command: {command}")
    print("   Status: Executing...")

    # ===== PRODUCTION: This would use Netmiko, NAPALM, or Nornir =====
    # from netmiko import ConnectHandler
    # device_config = {
    #     'device_type': 'cisco_ios',
    #     'host': device,
    #     'username': os.getenv('NET_USER'),
    #     'password': os.getenv('NET_PASS'),
    # }
    # connection = ConnectHandler(**device_config)
    # output = connection.send_command(command)
    # connection.disconnect()
    # return output

    # For demonstration, simulate realistic output
    simulated_output = f"""
{command} on {device}
========================================
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     192.168.1.1     YES NVRAM  up                    up
GigabitEthernet0/1     10.0.0.1        YES NVRAM  up                    up
GigabitEthernet0/2     10.0.1.1        YES NVRAM  up                    up
Loopback0              192.168.100.1   YES NVRAM  up                    up
========================================
Command executed successfully on {device}
"""

    print("   Status: ✅ Success")
    return simulated_output.strip()


@tool
def check_device_reachability(device: str) -> str:
    """
    Check if a network device is reachable (ping test).

    Use this tool to verify device connectivity before attempting to run commands.
    This is useful for troubleshooting and validating device availability.

    Args:
        device: Device hostname or IP address

    Returns:
        Reachability status with latency information
    """
    # In production, this would actually ping the device:
    # import subprocess
    # result = subprocess.run(['ping', '-c', '4', device], capture_output=True)
    # ...

    # Simulated ping result
    return f"✅ Device {device} is reachable\n   RTT: min=2.1ms, avg=3.5ms, max=5.2ms\n   Packet loss: 0%"


@tool
def get_device_info(device: str) -> str:
    """
    Retrieve basic information about a network device.

    Use this to get device type, model, software version, uptime, and other
    system information without needing to know specific show commands.

    Args:
        device: Device hostname or IP address

    Returns:
        Comprehensive device information
    """
    # In production, this would gather actual device info
    # Often combines: show version, show inventory, show system

    info = f"""
Device Information for {device}
========================================
Hostname:      {device}
Device Type:   Cisco IOS Router
Model:         ISR4331/K9
IOS Version:   15.6(2)T
Serial Number: FDO12345678
Uptime:        45 days, 12 hours, 34 minutes
CPU Usage:     15% (5 min average)
Memory:        512MB total, 340MB free
Flash:         4096MB total, 2048MB free
========================================
"""
    return info.strip()


@tool
def get_interface_status(device: str, interface: str | None = None) -> str:
    """
    Get status of network interfaces on a device.

    Args:
        device: Device hostname or IP address
        interface: Specific interface (e.g., "GigabitEthernet0/0") or None for all

    Returns:
        Interface status information
    """
    if interface:
        output = f"""
Interface: {interface} on {device}
========================================
Status:                up
Protocol:              up
IP Address:            192.168.1.1/24
Speed:                 1000 Mbps
Duplex:                Full
MTU:                   1500
Packets In/Out:        1234567 / 987654
Errors:                0
========================================
"""
    else:
        output = f"""
All Interfaces on {device}
========================================
Interface          Status  Protocol  IP Address      Speed
Gi0/0              up      up        192.168.1.1     1000Mbps
Gi0/1              up      up        10.0.0.1        1000Mbps
Gi0/2              down    down      unassigned      auto
Lo0                up      up        192.168.100.1   N/A
========================================
"""
    return output.strip()


# ===== TESTING =====
if __name__ == "__main__":
    print("=" * 60)
    print("Network Tools Test Suite")
    print("=" * 60)

    # Test 1: Safe command
    print("\n[Test 1] Safe Command Execution")
    print("-" * 60)
    result = run_network_command.invoke({
        "device": "switch-core-01",
        "command": "show ip interface brief",
    })
    print(result)

    # Test 2: Unsafe command (should be blocked)
    print("\n[Test 2] Unsafe Command (Should Block)")
    print("-" * 60)
    result = run_network_command.invoke({
        "device": "switch-core-01",
        "command": "reload in 5",
    })
    print(result)

    # Test 3: Another unsafe command
    print("\n[Test 3] Another Unsafe Command (Should Block)")
    print("-" * 60)
    result = run_network_command.invoke({
        "device": "router-edge-01",
        "command": "conf t",
    })
    print(result)

    # Test 4: Device reachability
    print("\n[Test 4] Device Reachability Check")
    print("-" * 60)
    result = check_device_reachability.invoke({"device": "router-edge-02"})
    print(result)

    # Test 5: Device information
    print("\n[Test 5] Device Information Retrieval")
    print("-" * 60)
    result = get_device_info.invoke({"device": "switch-access-05"})
    print(result)

    # Test 6: Interface status
    print("\n[Test 6] Interface Status")
    print("-" * 60)
    result = get_interface_status.invoke({
        "device": "router-core-01",
        "interface": "GigabitEthernet0/0",
    })
    print(result)

    print("\n" + "=" * 60)
    print("All Tests Complete")
    print("=" * 60)

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Use a more capable model for tool calling
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Permission database - simulates user access control
user_permissions = {
    "user_admin": ["read", "write", "delete"],
    "user_standard": ["read", "write"],
    "user_guest": ["read"],
}


@tool
def delete_file(user_id: str, filename: str) -> str:
    """Delete a file if user has permission. Requires 'delete' permission."""
    # Check if user exists
    permissions = user_permissions.get(user_id, [])

    # Validate permission
    if "delete" not in permissions:
        return f"❌ Permission denied: {user_id} cannot delete files (requires 'delete' permission)"

    # Validate filename extension
    if not filename.endswith((".txt", ".log")):
        return (
            f"❌ Invalid file type: can only delete .txt or .log files, got {filename}"
        )

    # In production, this would actually delete the file
    return f"✅ Successfully deleted {filename} (user: {user_id})"


@tool
def write_file(user_id: str, filename: str, content: str) -> str:
    """Write to a file if user has permission. Requires 'write' permission."""
    # Check if user exists
    permissions = user_permissions.get(user_id, [])

    # Validate permission
    if "write" not in permissions:
        return f"❌ Permission denied: {user_id} cannot write files (requires 'write' permission)"

    # Validate content length
    if len(content) > 1000:
        return f"❌ Content too large: maximum 1000 characters, got {len(content)}"

    # In production, this would write the file
    return f"✅ Successfully wrote {len(content)} characters to {filename} (user: {user_id})"


@tool
def read_file(user_id: str, filename: str) -> str:
    """Read a file if user has permission. Requires 'read' permission."""
    # Check if user exists
    permissions = user_permissions.get(user_id, [])

    # Validate permission
    if "read" not in permissions:
        return f"❌ Permission denied: {user_id} cannot read files (requires 'read' permission)"

    # In production, this would read the actual file
    return f"✅ File contents of {filename}: [Sample content] (user: {user_id})"


# Create agent with conditional tools
tools = [delete_file, write_file, read_file]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a file management assistant. Always check user permissions before performing operations.",
)

# Test scenarios demonstrating conditional tool execution
print("=" * 60)
print("CONDITIONAL TOOL EXECUTION DEMONSTRATION")
print("=" * 60)

# Scenario 1: Admin user (has all permissions)
print("\n--- Scenario 1: Admin User ---")
print("User: user_admin (permissions: read, write, delete)")
result1 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Delete the file 'old_data.txt' for user_admin"}
    ]
})
print(f"AI: {result1['messages'][-1].content}\n")

# Scenario 2: Standard user trying to delete (no permission)
print("--- Scenario 2: Standard User (No Delete Permission) ---")
print("User: user_standard (permissions: read, write)")
result2 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Delete the file 'report.txt' for user_standard"}
    ]
})
print(f"AI: {result2['messages'][-1].content}\n")

# Scenario 3: Guest user trying to write (no permission)
print("--- Scenario 3: Guest User (No Write Permission) ---")
print("User: user_guest (permissions: read)")
result3 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Write 'Hello World' to 'notes.txt' for user_guest"}
    ]
})
print(f"AI: {result3['messages'][-1].content}\n")

# Scenario 4: Invalid file type
print("--- Scenario 4: Invalid File Type ---")
print("User: user_admin (trying to delete .exe file)")
result4 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Delete the file 'program.exe' for user_admin"}
    ]
})
print(f"AI: {result4['messages'][-1].content}\n")

# Scenario 5: Successful operation
print("--- Scenario 5: Successful Write Operation ---")
print("User: user_standard (has write permission)")
result5 = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Write 'Meeting notes from today' to 'notes.txt' for user_standard",
        }
    ]
})
print(f"AI: {result5['messages'][-1].content}\n")

print("=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("1. Tools validate inputs BEFORE executing expensive operations")
print("2. Permission checks prevent unauthorized actions")
print("3. Type validation ensures data integrity")
print("4. Graceful error messages guide users to correct usage")
print("5. Conditional logic makes tools production-ready")

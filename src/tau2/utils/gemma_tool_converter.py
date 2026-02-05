"""
Convert OpenAI JSON schema tools to Gemma Python signature format.
Based on: https://www.philschmid.de/gemma-function-calling
"""

import json
from typing import Any, Optional


def json_schema_to_python_signature(tool: dict) -> str:
    """
    Convert OpenAI JSON schema tool definition to Python function signature.

    Args:
        tool: OpenAI tool schema in format {"type": "function", "function": {...}}

    Returns:
        Python function signature as string with type hints and docstring
    """
    func = tool.get("function", {})
    name = func.get("name", "unknown")
    description = func.get("description", "")
    parameters = func.get("parameters", {})

    # Extract parameter info
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    # Build parameter list with type hints
    params = []
    for param_name, param_info in properties.items():
        param_type = _json_type_to_python_type(param_info)

        # Add Optional[] wrapper if not required
        if param_name not in required:
            param_type = f"Optional[{param_type}]"

        params.append(f"{param_name}: {param_type}")

    params_str = ", ".join(params) if params else ""

    # Build docstring
    docstring_lines = [f'    """{description}']

    if properties:
        docstring_lines.append("")
        docstring_lines.append("    Args:")
        for param_name, param_info in properties.items():
            param_desc = param_info.get("description", "")
            docstring_lines.append(f"        {param_name}: {param_desc}")

    docstring_lines.append('    """')
    docstring = "\n".join(docstring_lines)

    # Build complete signature
    signature = f"def {name}({params_str}) -> dict:\n{docstring}"

    return signature


def _json_type_to_python_type(param_info: dict) -> str:
    """Convert JSON schema type to Python type hint."""
    json_type = param_info.get("type", "any")

    # Handle enums
    if "enum" in param_info:
        enum_values = param_info["enum"]
        # Return literal type hint
        return f"Literal[{', '.join(repr(v) for v in enum_values)}]"

    # Handle arrays
    if json_type == "array":
        items = param_info.get("items", {})
        item_type = _json_type_to_python_type(items) if items else "Any"
        return f"list[{item_type}]"

    # Handle objects
    if json_type == "object":
        return "dict"

    # Basic type mapping
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "null": "None",
    }

    return type_map.get(json_type, "Any")


def convert_tools_to_gemma_format(tools: list[dict]) -> str:
    """
    Convert list of OpenAI tools to Gemma-compatible Python signatures.

    Args:
        tools: List of OpenAI tool schemas

    Returns:
        Formatted string with all Python function signatures
    """
    signatures = []

    for tool in tools:
        signature = json_schema_to_python_signature(tool)
        signatures.append(signature)

    # Join with double newlines
    return "\n\n".join(signatures)


def create_gemma_system_prompt_with_tools(system_prompt: str, tools: list[dict]) -> str:
    """
    Create Gemma-compatible system prompt with tools as Python signatures.

    Args:
        system_prompt: Original system prompt
        tools: List of OpenAI tool schemas

    Returns:
        Combined prompt with instructions and Python function signatures
    """
    tool_signatures = convert_tools_to_gemma_format(tools)

    tools_instruction = f"""
# Available Tools

The following Python functions are available for you to call. To use them, respond with a code block using ```tool_code``` syntax.

{tool_signatures}

## Tool Calling Instructions

When you need to call a function:
1. Wrap your function call in ```tool_code``` markdown code blocks
2. Call the function with proper Python syntax: function_name(arg1=value1, arg2=value2)
3. Do NOT add any other text when making a tool call
4. After receiving tool output in ```tool_output``` blocks, you can call more tools or respond to the user

Example:
```tool_code
search_flights(origin="JFK", destination="LAX", date="2024-05-20")
```

Only call ONE function at a time. Wait for the tool output before calling another function or responding.
"""

    return system_prompt + "\n\n" + tools_instruction

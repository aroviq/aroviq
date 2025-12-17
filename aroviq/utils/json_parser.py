import json
import re
import ast
from typing import Any, Dict

def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Parses JSON from a string that might contain Markdown code blocks
    or python-style dict syntax. Robustly handles trailing commas.
    
    Args:
        text (str): The string output from an LLM.
        
    Returns:
        Dict[str, Any]: The parsed dictionary.
        
    Raises:
        ValueError: If parsing fails completely.
    """
    if not text:
        raise ValueError("Empty input string.")

    # 1. Strip Markdown code fences
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # 2. Extract JSON payload (find first { and last })
    match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_braces:
        text = match_braces.group(1)
    
    # 3. Handle trailing commas (common LLM error)
    # This regex looks for a comma followed by closing brace/bracket and removes it.
    # It removes ,} and ,] patterns possibly separated by whitespace.
    text = re.sub(r",\s*([\]}])", r"\1", text)

    # 4. Try strict JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 5. Try ast.literal_eval (handling python-style dicts)
    try:
        # Note: ast.literal_eval handles trailing commas in Python syntax naturally
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
        
    # 6. Fallback: specific fix for single quotes acting as JSON
    try:
        # Replace single quotes with double quotes (rough heuristic)
        # This is risky if string contains content with quotes, but useful for desperate recovery
        fixed_text = text.replace("'", '"')
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse JSON from text: {text}")

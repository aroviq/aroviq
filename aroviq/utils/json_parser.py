import json
import re
from typing import Any


def parse_llm_json(text: str, *, max_chars: int = 20000) -> dict[str, Any]:
    """
    Parses JSON from a string that might contain Markdown code blocks
    or loose formatting. Strictly enforces a top-level object and bounded size.
    
    Args:
        text (str): The string output from an LLM.
        
    Returns:
        Dict[str, Any]: The parsed dictionary.
        
    Raises:
        ValueError: If parsing fails completely.
    """
    if not text:
        raise ValueError("Empty input string.")

    if len(text) > max_chars:
        raise ValueError("LLM response exceeds maximum allowed length.")

    # 1. Strip Markdown code fences
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    candidate = text.strip()

    # 2. Find the first JSON object
    start = candidate.find("{")
    if start == -1:
        raise ValueError("LLM response does not contain a JSON object.")

    candidate = candidate[start:]

    # 3. Handle trailing commas (common LLM error)
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON from text: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON was not an object.")

    trailing = candidate[end:].strip()
    if trailing:
        raise ValueError("Extra data detected after JSON object.")

    return obj

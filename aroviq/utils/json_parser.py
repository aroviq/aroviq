import json
import re
from typing import Any

from aroviq.utils.text import strip_code_fence

def parse_llm_json(text: str, *, max_chars: int = 20000) -> dict[str, Any]:
    """
    Parses JSON from a string that might contain Markdown code blocks
    or loose formatting. Strictly enforces a top-level object and bounded size.

    Intentionally omitted fallbacks
    --------------------------------
    Earlier versions of this function included two additional recovery steps
    that have since been removed:

    **Step 5 — ``ast.literal_eval``**: Accepted Python dict syntax (e.g.
    ``{'key': 'value'}``).  This is a superset of JSON that allows arbitrary
    Python literals.  An attacker who controls LLM output could craft a payload
    that parses as a valid Python literal but not as JSON, yielding a dict whose
    ``approved`` key resolves to ``True`` through Python truthiness rules rather
    than strict boolean comparison.  Removed.

    **Step 6 — ``text.replace("'", '"')``**: A blanket single-to-double-quote
    substitution.  The original comment even flagged it: *"This is risky if
    string contains content with quotes."*  An attacker could embed a single-
    quoted string whose content, after substitution, produces
    ``"approved": true`` where the original intent was ``"approved": false``.
    Any fallback that rewrites untrusted content before parsing it is a
    semantic-mangling attack surface.  Removed.

    Both steps increased parser resilience at the cost of correctability —
    they made it harder to know *what* the LLM actually said.  For a security
    verifier this is the wrong trade-off.  A parse failure that blocks a step
    (fail-closed) is safer than a recovery that silently mangles the verdict.

    .. todo::
        The correct long-term fix is to eliminate free-form JSON parsing
        entirely by using structured outputs (e.g. OpenAI's
        ``response_format={"type": "json_schema", ...}`` or Gemini's
        ``response_mime_type="application/json"`` with a schema) so the LLM
        is constrained to emit valid JSON at the model level, not patched
        post-hoc in the client.  Track in GitHub issue: structured-output
        migration.

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
    text = strip_code_fence(text)

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
        snippet = candidate[:120].replace("\n", " ")
        raise ValueError(
            f"Could not parse JSON from text (starts with: {snippet!r}): {exc}"
        ) from exc

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON was not an object.")

    trailing = candidate[end:].strip()
    if trailing:
        raise ValueError("Extra data detected after JSON object.")

    return obj

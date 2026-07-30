"""JSON-schema helpers for providers without native Pydantic support.

Purpose:
    Let providers that accept only a raw JSON schema — or no schema at all —
    still satisfy the ``parse()`` half of the ``AIProvider`` contract.

Why this exists:
    Anthropic's SDK takes a Pydantic model directly and hands back a validated
    instance. Gemini and the OpenAI-compatible endpoints do not: they take a JSON
    schema in a dialect that is *nearly* the one Pydantic emits, and return a
    string that is *usually* valid JSON.

    "Nearly" and "usually" are the whole problem, and each gets a function here:
    one to normalise the schema into what those APIs accept, one to recover a
    model instance from a response that may be wrapped in prose or a code fence.

Dependencies:
    pydantic

Future improvements:
    * Cache normalised schemas; they are stable per model class.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

#: Matches a fenced code block, with or without a language tag. Models wrap JSON
#: in one often enough that stripping it is worth doing before giving up.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

#: Schema keywords Gemini rejects outright. Pydantic emits several of them for
#: ordinary models, so they are removed rather than avoided upstream.
_UNSUPPORTED_KEYS = frozenset(
    {"additionalProperties", "$schema", "discriminator", "examples", "const", "default"}
)


def flatten_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a JSON schema with ``$ref`` indirection resolved.

    Nested models make Pydantic emit ``$defs`` plus ``$ref`` pointers. Gemini
    does not follow refs, so a nested model would arrive as an empty object and
    the response would validate against nothing useful. Inlining is not an
    optimisation here — without it, structured output silently degrades.

    Args:
        model: The Pydantic model to describe.

    Returns:
        A self-contained schema with no ``$ref`` or ``$defs``.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    inlined: dict[str, Any] = _inline(schema, definitions)
    return inlined


def gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a schema Gemini's ``responseSchema`` accepts.

    Args:
        model: The Pydantic model to describe.

    Returns:
        A flattened schema with unsupported keywords stripped.
    """
    stripped: dict[str, Any] = _strip_unsupported(flatten_schema(model))
    return stripped


def parse_model[ModelT: BaseModel](text: str, model: type[ModelT]) -> ModelT | None:
    """Recover a model instance from a raw response.

    Tolerant on purpose. A provider asked for JSON may return it fenced, or with
    a sentence in front, or with trailing commentary. Failing on any of those
    would make structured output unreliable for reasons that have nothing to do
    with the model's actual answer.

    Args:
        text: The raw response text.
        model: The model to validate against.

    Returns:
        A validated instance, or ``None`` when nothing usable was found.
    """
    for candidate in _json_candidates(text):
        try:
            return model.model_validate_json(candidate)
        except ValidationError:
            continue
        except ValueError:
            continue
    return None


def schema_instruction(model: type[BaseModel]) -> str:
    """Build a prompt fragment describing the required output shape.

    Used with providers that have no schema parameter at all — the schema
    becomes part of the instruction instead.

    Args:
        model: The model the response must match.

    Returns:
        Text to append to the system prompt.
    """
    schema = json.dumps(flatten_schema(model), indent=2)
    return (
        "Reply with a single JSON object and nothing else — no prose, no code "
        f"fence, no explanation. It must match this schema:\n\n{schema}"
    )


# -- internals -------------------------------------------------------------


def _inline(node: Any, definitions: dict[str, Any]) -> Any:
    """Replace ``$ref`` pointers with the definitions they name.

    Args:
        node: The schema fragment being walked.
        definitions: The ``$defs`` block from the original schema.

    Returns:
        The fragment with refs resolved.
    """
    if isinstance(node, dict):
        if ref := node.get("$ref"):
            name = str(ref).rsplit("/", 1)[-1]
            resolved = definitions.get(name, {})
            merged = {**_inline(resolved, definitions)}
            # Keep sibling keys such as `description`, which sit alongside $ref.
            merged.update({k: _inline(v, definitions) for k, v in node.items() if k != "$ref"})
            return merged
        return {key: _inline(value, definitions) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline(item, definitions) for item in node]
    return node


def _strip_unsupported(node: Any) -> Any:
    """Remove schema keywords Gemini rejects.

    Args:
        node: The schema fragment being walked.

    Returns:
        The fragment with unsupported keywords removed.
    """
    if isinstance(node, dict):
        return {
            key: _strip_unsupported(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_KEYS
        }
    if isinstance(node, list):
        return [_strip_unsupported(item) for item in node]
    return node


def _json_candidates(text: str) -> list[str]:
    """Extract plausible JSON payloads from a response, best first.

    Args:
        text: The raw response.

    Returns:
        Candidate JSON strings to try validating.
    """
    stripped = text.strip()
    candidates = [stripped]

    if fenced := _FENCE.search(stripped):
        candidates.insert(0, fenced.group(1))

    # Outermost braces: catches "Here is the result: {...}" and trailing notes.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    return candidates

"""Prompt helpers for arithmetic tool-use experiments."""

from __future__ import annotations

import json
from typing import Any, Dict


ARITHMETIC_SYSTEM_PROMPT_TEMPLATE = (
    "You are an arithmetic agent that must solve each question with the provided tools.\n"
    "Never use internal knowledge or mental math for the final answer.\n"
    "Only use information returned by tools in this episode.\n"
    "If the tools are insufficient or uncertain, answer Unknown.\n\n"
    "You have access to the following arithmetic tools:\n\n"
    "{tool_descriptions}\n\n"
    "You must manage your tool usage within the given time and cost budgets.\n"
    "At each step, you may either:\n"
    "  1. Call a tool using structured JSON (preferred):\n"
    "     {{\"thought\": \"...\", \"tool_name\": \"<tool>\", \"arguments\": {{\"a\": 1, \"b\": 2}}}}\n"
    "  2. Give a final answer:\n"
    "     Action: Answer is \\\\boxed{{ANS}}.\n\n"
    "ANS must be a value derived from tool outputs only, or Unknown.\n"
    "Be strategic about which tool variant to use (verified vs old vs verylightweight) "
    "based on remaining budget.\n"
)

ARITHMETIC_RESPONSE_FORMAT_HINT = (
    "Could not parse your response. Please respond with either:\n"
    "  {\"thought\": \"...\", \"tool_name\": \"<tool>\", \"arguments\": {...}}\n"
    "  Action: tool_name(a=<val>, b=<val>)\n"
    "  Action: Answer is \\\\boxed{ANS}."
)

ARITHMETIC_FORCE_FINAL_ANSWER_HINT = (
    "Time/cost budget is exhausted. Stop using tools immediately.\n"
    "Use only previously observed tool outputs from this episode.\n"
    "Do not use internal knowledge, guessing, or mental math.\n"
    "Respond with exactly one line in this format:\n"
    "Action: Answer is \\boxed{ANS}.\n"
    "If unknown, use: Action: Answer is \\boxed{Unknown}."
)


def _to_prompt_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def format_arithmetic_tool_descriptions(tool_registry: Dict[str, Any]) -> str:
    """Build prompt text for all arithmetic tools with metadata."""
    seen: Dict[str, str] = {}
    for name, tool in tool_registry.items():
        tool_name = getattr(tool, "name", name)
        if tool_name in seen:
            continue

        desc = getattr(tool, "description", "")
        inputs = getattr(tool, "inputs", {})
        output_type = getattr(tool, "output_type", "any")
        output_quality = getattr(tool, "output_quality", "unknown")

        execution_time_mu = getattr(tool, "execution_time_mu", None)
        default_execution_time = getattr(tool, "default_execution_time", 0)
        execution_time_sigma = getattr(tool, "execution_time_sigma", None)
        execution_time_min = getattr(tool, "execution_time_min", None)
        execution_time_max = getattr(tool, "execution_time_max", None)

        mu = execution_time_mu if execution_time_mu is not None else default_execution_time
        sigma = execution_time_sigma if execution_time_sigma is not None else 0
        support_min = execution_time_min if execution_time_min is not None else 0
        support_max = execution_time_max if execution_time_max is not None else 0

        seen[tool_name] = (
            f"- {tool_name}: {desc}\n"
            f"    Takes inputs: {_to_prompt_value(inputs)}\n"
            f"    Returns an output of type: {_to_prompt_value(output_type)}\n"
            f"    output quality: {_to_prompt_value(output_quality)}\n"
            "    tool execution time: sampled per call.\n"
            f"      - mean (mu): {_to_prompt_value(mu)} seconds\n"
            f"      - stddev (sigma): {_to_prompt_value(sigma)} seconds\n"
            f"      - support: [{_to_prompt_value(support_min)}, {_to_prompt_value(support_max)}]"
        )

    if not seen:
        return "- (no tools available)"
    return "\n".join(seen[tool_name] for tool_name in sorted(seen.keys()))


def build_arithmetic_system_prompt(tool_registry: Dict[str, Any]) -> str:
    """Render full arithmetic system prompt from a tool registry."""
    return ARITHMETIC_SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=format_arithmetic_tool_descriptions(tool_registry)
    )

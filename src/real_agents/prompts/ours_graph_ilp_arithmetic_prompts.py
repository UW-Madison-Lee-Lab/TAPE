"""Prompt templates and JSON schemas for Arithmetic OursGraphILP agent.

Graph structure:
  - Nodes: computation states (intermediate results accumulated so far).
  - Edges: tool calls (tool_name + arguments).
  - Goal node: state that contains the final answer.
  - Constrained decoding: tool_name is an enum of available tool names.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helper: dynamically build tool-name enum from a registry
# ---------------------------------------------------------------------------

def _collect_tool_names(tool_registry: Dict[str, Any]) -> List[str]:
    """Return sorted, unique tool names from the registry."""
    seen = set()
    for _key, tool in tool_registry.items():
        name = getattr(tool, "name", None) or _key
        seen.add(name)
    return sorted(seen)


def _tool_description_block(tool_registry: Dict[str, Any]) -> str:
    """One-line-per-tool description block for prompts."""
    seen: Dict[str, str] = {}
    for _key, tool in tool_registry.items():
        name = getattr(tool, "name", _key)
        if name in seen:
            continue
        desc = getattr(tool, "description", "")
        quality = getattr(tool, "output_quality", "?")
        mu_time = getattr(tool, "execution_time_mu", getattr(tool, "default_execution_time", "?"))
        mu_cost = getattr(tool, "cost_mu", getattr(tool, "default_cost", "?"))
        inputs = getattr(tool, "inputs", {})
        input_keys = ", ".join(inputs.keys()) if isinstance(inputs, dict) else "a, b"
        seen[name] = (
            f"  - {name}({input_keys}): {desc}  "
            f"[quality={quality}, avg_time={mu_time}, avg_cost={mu_cost}]"
        )
    return "\n".join(seen[n] for n in sorted(seen))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_INSTRUCTION_TEMPLATE = """\
You are an arithmetic problem-solving planner.

You have access to the following arithmetic tools (each with different accuracy, \
execution time, and monetary cost):

{tool_descriptions}

## Instructions
Given a question and available budgets (some budget fields may be omitted), plan {num_plans} diverse sequences of \
tool calls that compute the answer.
Each plan is a sequence of tool calls like:
  Step 1: tool_name(a=<val>, b=<val>)  → expected result
  Step 2: tool_name(a=<prev_result>, b=<val>)  → expected result
  ...
  Final Answer: <answer>

Guidelines:
- Use cheaper/faster tools when the budget is tight; use verified tools when the budget allows.
- Each plan MUST stay within the enabled budgets.
- Plans should be diverse: vary tool variants (verified / old / verylightweight) \
and computation order.
- Provide exactly {num_plans} plans labeled "Plan 1:" … "Plan {num_plans}:".
"""

GRAPH_STEP1_SYSTEM_TEMPLATE = """\
Simulate arithmetic tool-use plans and produce per-plan step sequences to \
build a computation graph.

Available tools:
{tool_descriptions}

## Instructions
- Simulate each plan step-by-step.
- Build each plan's step sequence as alternating entries: node → action → node → …
- A **node** entry has: node_id, thought, observation.
  - observation: a JSON string summarizing the computation state so far \
(e.g., intermediate results, remaining budgets).
  - If a budget dimension is not provided, omit that field from observation \
    (e.g., omit time_remaining when no time budget is given).
- An **action** entry has: tool_name, arguments (object with the tool's \
input keys), expected_result (string).
- Node ids must be unique across all plans.
- Preserve ALL steps from every plan.
- The first node of each plan shares the same initial state (the question + \
full budgets).
- tool_name MUST be one of the available tool names listed above.

Return JSON only:
{{
  "plan_sequences": [
    {{
      "plan_id": "plan1",
      "steps": [
        {{
          "node_id": "n0",
          "thought": "Initial state: no computation done yet.",
          "observation": "question: ... | results: [] | time_remaining: ... | cost_remaining: ..."
        }},
        {{
          "tool_name": "add_verified",
          "arguments": {{"a": 3, "b": 5}},
          "expected_result": "8"
        }},
        {{
          "node_id": "n1",
          "thought": "After add_verified(3,5)=8.",
          "observation": "question: ... | results: [8] | time_remaining: ... | cost_remaining: ..."
        }}
      ]
    }}
  ]
}}
"""

GRAPH_STEP2_INSTRUCTIONS = """\
## Instructions
- After all plan sequences are built, merge identical nodes when the \
observation text matches exactly.
- Preserve ALL tool calls from every plan (do not drop steps).
- Construct the full graph (nodes + edges) from the merged nodes.
- Generate a thought for each edge based on the tool call and its effect.
- Mark goal nodes (nodes where the answer is known) with "is_goal": true.
- JSON keys must appear in this order: reasoning, merge_log, full_graph.

Return JSON only:
{
  "reasoning": "overall reasoning for node merging and graph construction",
  "merge_log": [
    {
      "reason": "same observation text",
      "kept_node": "n0",
      "merged_nodes": ["n0_plan2", ...]
    }
  ],
  "full_graph": {
    "nodes": [
      {
        "id": "n0",
        "observation": "question: ... | results: [] | ...",
        "is_start": true,
        "is_goal": false
      }
    ],
    "edges": [
      {
        "from": "n0",
        "to": "n1",
        "thought": "compute 3+5 using add_verified",
        "tool_name": "add_verified",
        "arguments": {"a": 3, "b": 5},
        "expected_result": "8"
      }
    ]
  }
}
"""

PLAN_USER_TEMPLATE = """\
Question: {question}
{budget_lines}
{remaining_lines}
Computation so far: {computation_history}
"""

GRAPH_STEP1_USER_TEMPLATE = """\
Question: {question}
{budget_lines}
{remaining_lines}
Computation so far: {computation_history}

Plans text:
{plans_text}"""

GRAPH_STEP2_USER_TEMPLATE = """\
Use the plan_sequences you just produced.
{instructions}"""

SCORE_NODES_SYSTEM = """\
You score all the computation states in the graph.
Higher score means closer to having the correct final answer.

Assign score 1.0 to goal states (where the final answer is determined).
Assign score -1.0 to dead-end states (budget exhausted with no answer, \
or unreachable goal).
All other states should have score 0.0.

Only use scores -1.0, 0.0, or 1.0 (no other values).
For each state, provide a short reasoning sentence before assigning its score.
Return JSON only:
{{
  "reasons": {{
    "n0": "reasoning for n0",
    ...
  }},
  "scores": {{
    "n0": 0.0,
    ...
  }}
}}
"""

SCORE_NODES_USER_TEMPLATE = """\
Question: {question}
{remaining_lines}
Available Transitions (Edges):
{edges}
Target States to Score:
{states}"""

REACT_INSTRUCTION_TEMPLATE = """\
You are an arithmetic agent that must solve each question with the provided tools.
Never use internal knowledge or mental math for the final answer.
Only use information returned by tools in this episode.

Available tools:
{tool_descriptions}

You must manage your tool usage within the given time and cost budgets.
At each step, you may either:
  1. Call a tool: Thought: <reasoning>
Action: tool_name(a=<val>, b=<val>)
  2. Give a final answer: Thought: <reasoning>
Action: Answer is \\boxed{{ANS}}.

{plan_section}
Be strategic about which tool variant to use (verified vs old vs verylightweight) \
based on remaining budget.
"""

SYSTEM_PROMPT_TEMPLATE = """\
{{ INSTRUCTION }}

{% if examples_text %}
{{ icl_prompt }}
{{ examples_text }}
{% endif %}
"""


# ---------------------------------------------------------------------------
# JSON-schema builders (constrained decoding)
# ---------------------------------------------------------------------------

def build_graph_step1_response_format(tool_names: List[str]) -> Dict[str, Any]:
    """Build OpenAI-style response_format for graph step 1 with tool-name enum."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "arithmetic_plan_sequences",
            "schema": {
                "type": "object",
                "properties": {
                    "plan_sequences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "plan_id": {"type": "string"},
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "oneOf": [
                                            # Node entry
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "node_id": {"type": "string"},
                                                    "thought": {"type": "string"},
                                                    "observation": {"type": "string"},
                                                },
                                                "required": ["node_id", "thought", "observation"],
                                                "additionalProperties": False,
                                            },
                                            # Action entry (tool call)
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "tool_name": {
                                                        "type": "string",
                                                        "enum": tool_names,
                                                    },
                                                    "arguments": {
                                                        "type": "object",
                                                    },
                                                    "expected_result": {"type": "string"},
                                                },
                                                "required": ["tool_name", "arguments", "expected_result"],
                                                "additionalProperties": False,
                                            },
                                        ]
                                    },
                                    "minItems": 1,
                                },
                            },
                            "required": ["plan_id", "steps"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                },
                "required": ["plan_sequences"],
                "additionalProperties": False,
            },
        },
    }


def build_graph_step1_response_format_claude(tool_names: List[str]) -> Dict[str, Any]:
    """Build Claude structured-output format for graph step 1."""
    return {
        "type": "object",
        "properties": {
            "plan_sequences": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "steps": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": ["node", "action"],
                                    },
                                    # node fields
                                    "node_id": {"type": "string"},
                                    "thought": {"type": "string"},
                                    "observation": {"type": "string"},
                                    # action fields
                                    "tool_name": {
                                        "type": "string",
                                        "enum": tool_names,
                                    },
                                    "arguments": {"type": "object"},
                                    "expected_result": {"type": "string"},
                                },
                                "required": ["kind"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["plan_id", "steps"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plan_sequences"],
        "additionalProperties": False,
    }


def build_graph_step2_response_format(tool_names: List[str]) -> Dict[str, Any]:
    """Build OpenAI-style response_format for graph step 2 with tool-name enum."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "arithmetic_graph",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "merge_log": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kept_node": {"type": "string"},
                                "merged_nodes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["kept_node", "merged_nodes", "reason"],
                            "additionalProperties": False,
                        },
                    },
                    "full_graph": {
                        "type": "object",
                        "properties": {
                            "nodes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "observation": {"type": "string"},
                                        "is_start": {"type": "boolean"},
                                        "is_goal": {"type": "boolean"},
                                    },
                                    "required": ["id", "observation", "is_start", "is_goal"],
                                    "additionalProperties": False,
                                },
                                "minItems": 1,
                            },
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from": {"type": "string"},
                                        "to": {"type": "string"},
                                        "thought": {"type": "string"},
                                        "tool_name": {
                                            "type": "string",
                                            "enum": tool_names,
                                        },
                                        "arguments": {"type": "object"},
                                        "expected_result": {"type": "string"},
                                    },
                                    "required": [
                                        "from", "to", "thought",
                                        "tool_name", "arguments", "expected_result",
                                    ],
                                    "additionalProperties": False,
                                },
                                "minItems": 1,
                            },
                        },
                        "required": ["nodes", "edges"],
                        "additionalProperties": False,
                    },
                },
                "required": ["reasoning", "merge_log", "full_graph"],
                "additionalProperties": False,
            },
        },
    }


def build_graph_step2_response_format_claude(tool_names: List[str]) -> Dict[str, Any]:
    """Build Claude structured-output format for graph step 2."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "merge_log": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kept_node": {"type": "string"},
                        "merged_nodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["kept_node", "merged_nodes", "reason"],
                    "additionalProperties": False,
                },
            },
            "full_graph": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "observation": {"type": "string"},
                                "is_start": {"type": "boolean"},
                                "is_goal": {"type": "boolean"},
                            },
                            "required": ["id", "observation", "is_start", "is_goal"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "thought": {"type": "string"},
                                "tool_name": {
                                    "type": "string",
                                    "enum": tool_names,
                                },
                                "arguments": {"type": "object"},
                                "expected_result": {"type": "string"},
                            },
                            "required": [
                                "from", "to", "thought",
                                "tool_name", "arguments", "expected_result",
                            ],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                },
                "required": ["nodes", "edges"],
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "merge_log", "full_graph"],
        "additionalProperties": False,
    }

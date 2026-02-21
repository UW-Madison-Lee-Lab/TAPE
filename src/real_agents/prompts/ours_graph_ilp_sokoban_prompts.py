"""Prompt templates and JSON schemas for Sokoban OursGraphILP agent."""

RULE = """## Sokoban rules (task + mechanics)
- Task objective: place every box onto goals. The puzzle is solved when all boxes are on goals.
- Action mechanics: each action moves the player exactly one cell in the chosen direction (U/D/L/R).
    - U: move up, e.g., player at (x, y) -> player at (x, y + 1)
    - D: move down, e.g., player at (x, y) -> player at (x, y - 1)
    - L: move left, e.g., player at (x, y) -> player at (x - 1, y)
    - R: move right, e.g., player at (x, y) -> player at (x + 1, y)
    - The action name must match the coordinate change (U increases y, D decreases y, R increases x, L decreases x).
- Walls: the player cannot move into a wall cell. If a wall occupies the destination cell, the action is invalid and the player does not move.
- Boxes: if the destination cell has a box, the player attempts to push it one cell further in the same direction.
    - The push succeeds only if the cell behind the box is empty floor or a goal.
    - If the cell behind the box is a wall or another box, the push is invalid and nothing moves.
    - A push is only possible when the player is on the cell immediately adjacent to the box from the opposite side of the push direction.
- The player cannot pull boxes and cannot push two boxes at once.
- Some pushes can create deadlocks (for example, pushing a box into a corner where it cannot reach any goal).

Examples (coordinate outcomes, using U/D/L/R only):
- Empty move, R: player at (x, y), no wall/box at (x + 1, y). Action R -> player at (x + 1, y); box on goal location unchanged (goals are unaffected by moves).
- Empty move, L: player at (x, y), no wall/box at (x - 1, y). Action L -> player at (x - 1, y); box on goal location unchanged (goals are unaffected by moves).
- Empty move, U: player at (x, y), no wall/box at (x, y + 1). Action U -> player at (x, y + 1); box on goal location unchanged (goals are unaffected by moves).
- Empty move, D: player at (x, y), no wall/box at (x, y - 1). Action D -> player at (x, y - 1); box on goal location unchanged (goals are unaffected by moves).
- Wall block, U: player at (x, y), wall at (x, y + 1). Action U -> invalid, player stays at (x, y); box on goal location unchanged.
- Wall block, D: player at (x, y), wall at (x, y - 1). Action D -> invalid, player stays at (x, y); box on goal location unchanged.
- Wall block, L: player at (x, y), wall at (x - 1, y). Action L -> invalid, player stays at (x, y); box on goal location unchanged.
- Wall block, R: player at (x, y), wall at (x + 1, y). Action R -> invalid, player stays at (x, y); box on goal location unchanged.
- Push succeeds, R: player at (x, y), box at (x + 1, y), no wall/box at (x + 2, y). Action R -> player at (x + 1, y), box moves to (x + 2, y); if (x + 2, y) is a goal, box on goal location becomes (x + 2, y), otherwise unchanged.
- Push succeeds, L: player at (x, y), box at (x - 1, y), no wall/box at (x - 2, y). Action L -> player at (x - 1, y), box moves to (x - 2, y); if (x - 2, y) is a goal, box on goal location becomes (x - 2, y), otherwise unchanged.
- Push succeeds, U: player at (x, y), box at (x, y + 1), no wall/box at (x, y + 2). Action U -> player at (x, y + 1), box moves to (x, y + 2); if (x, y + 2) is a goal, box on goal location becomes (x, y + 2), otherwise unchanged.
- Push succeeds, D: player at (x, y), box at (x, y - 1), no wall/box at (x, y - 2). Action D -> player at (x, y - 1), box moves to (x, y - 2); if (x, y - 2) is a goal, box on goal location becomes (x, y - 2), otherwise unchanged.
- Push blocked by wall, R: player at (x, y), box at (x + 1, y), wall at (x + 2, y). Action R -> invalid, player stays at (x, y), box stays at (x + 1, y); box on goal location unchanged.
- Push blocked by wall, L: player at (x, y), box at (x - 1, y), wall at (x - 2, y). Action L -> invalid, player stays at (x, y), box stays at (x - 1, y); box on goal location unchanged.
- Push blocked by wall, U: player at (x, y), box at (x, y + 1), wall at (x, y + 2). Action U -> invalid, player stays at (x, y), box stays at (x, y + 1); box on goal location unchanged.
- Push blocked by wall, D: player at (x, y), box at (x, y - 1), wall at (x, y - 2). Action D -> invalid, player stays at (x, y), box stays at (x, y - 1); box on goal location unchanged.
- Push blocked by box, R: player at (x, y), box at (x + 1, y), another box at (x + 2, y). Action R -> invalid, player stays at (x, y), boxes stay at (x + 1, y) and (x + 2, y); box on goal location unchanged.
- Push blocked by box, L: player at (x, y), box at (x - 1, y), another box at (x - 2, y). Action L -> invalid, player stays at (x, y), boxes stay at (x - 1, y) and (x - 2, y); box on goal location unchanged.
- Push blocked by box, U: player at (x, y), box at (x, y + 1), another box at (x, y + 2). Action U -> invalid, player stays at (x, y), boxes stay at (x, y + 1) and (x, y + 2); box on goal location unchanged.
- Push blocked by box, D: player at (x, y), box at (x, y - 1), another box at (x, y - 2). Action D -> invalid, player stays at (x, y), boxes stay at (x, y - 1) and (x, y - 2); box on goal location unchanged."""

REACT_INSTRUCTION = \
f"""Interact with Sokoban environment to solve a task (placing every box onto goals.)

{{sokoban_rule}}

## Instruction
You are the player in a Sokoban environment, and your goal is to place every box on a goal within a limited number of actions (within step remaining). That means you need to make the same number of boxes on goals by placing all boxes onto goals. If a box is on a goal, it is considered satisfied.

At each turn, you will receive the current observation.
Observation format (all coordinates are (x, y)):
- wall location: (x1, y1), (x2, y2), ...
- player location: (x, y)
- box location: (x3, y3), ...
- goal location: (x4, y4), ...
- box on goal location: (x5, y5), ...
- Step remaining: <steps_remaining>
The "box location" list includes only boxes not on goals. The "goal location" list includes all goals.
You may choose between two outputs: "Thought" or "Action".

When you choose "Thought", you must:
The initial full plan is given, so you compare the current observation and the full plan to identify the current state in the given plan and follow the given plan if it matches the current observation.
If there is a mismatch between plan and observation, you can replan and generate the action as follows:
Replan the full remaining path so that all boxes reach goals within the remaining steps, using the Sokoban rules (task + mechanics). From that full plan, explicitly predict only the next 1 step: how the immediate action will change the player location, box location, and box on goal location. Based on that planning and the current observation, decide the immediate next action to take.
IMPORTANT:
    - Your Thought MUST match the given observation exactly. No hallucination about positions or adjacency is allowed.
    - If a push is feasible immediately, you MUST choose the move that pushes.
    - If you planned a push in the previous Thought and the current observation still allows that push, you MUST continue and execute it.
    - Do NOT rewrite the plan from scratch unless the environment changed and the old plan became infeasible.
    - Always compute the next coordinate explicitly from the action (U/D/L/R) before describing it; do not rely on visual grid assumptions.
    - If the destination cell has a box, check the cell behind it: if it is empty or a goal, the move is a valid push (not invalid).

Your output must follow exactly:
Thought: <your reasoning>
Action: <U|D|L|R>
"""

PLAN_INSTRUCTION = \
f"""Interact with Sokoban environment to solve a task (placing every box onto goals.)

{{sokoban_rule}}

## Instruction
You are the player in a Sokoban environment, and your goal is to place every box on a goal within a limited number of actions (within step remaining). That means you need to make the same number of boxes on goals by placing all boxes onto goals. If a box is on a goal, it is considered satisfied.

At each turn, you will receive the current observation.
Observation format (all coordinates are (x, y)):
- wall location: (x1, y1), (x2, y2), ...
- player location: (x, y)
- box location: (x3, y3), ...
- goal location: (x4, y4), ...
- box on goal location: (x5, y5), ...
- Step remaining: <steps_remaining>
The "box location" list includes only boxes not on goals. The "goal location" list includes all goals.

You are in planning mode. Using the Sokoban rules, plan the full solution (overall path) so that all boxes reach goals within the remaining steps.
Generate {{num_plans}} diverse and valid plans.
For each plan, you must **precise** reason based on the location of player, box, goal, and the wall, and the Sokoban rules (task + mechanics). Then, generate the full solution (overall path) so that all boxes reach goals within the remaining steps, using the Sokoban rules (task + mechanics).
Based on this plan, you generate the full sequence of actions.
Generate as **diverse** as possible while maintaining success within given remaining steps.
Each plan MUST have a full action sequence (at most remaining steps of actions).
Provide exactly {{num_plans}} plans labeled "Plan 1:" ,..., "Plan {{num_plans}}:".

IMPORTANT: Use the action-coordinate rules exactly. Actions update the player location: U (moving up) moves the player (x, y) -> (x, y+1), D (moving down) moves the player (x, y) -> (x, y-1), R (moving right) moves the player (x, y) -> (x+1, y), L (moving left) moves the player (x, y) -> (x-1, y). Always align your verbal directions (up/down/left/right) with these coordinate changes. When describing relative positions, use the coordinate conventions: "above" means larger y, "below" means smaller y, "right" means larger x, "left" means smaller x. Validate each step in the plan by explicitly computing the next (x, y) from the action; if the destination cell is a box, check the cell behind it and treat the move as a valid push if that cell is empty or a goal, otherwise the move is invalid and must not appear in the action sequence.
If the player pushes a box, the box will move one cell in the same direction as the player unless blocked by a wall or another box.
After any move (including a push), the player and any box must occupy different cells; they can never share the same location.
"""

GRAPH_STEP1_SYSTEM = \
f"""Simulate Sokoban plans, produce per-plan step sequences and generate the graph.

{{sokoban_rule}}

IMPORTANT: Use the action-coordinate rules exactly. Actions update the player location: U (moving up) moves the player (x, y) -> (x, y+1), D (moving down) moves the player (x, y) -> (x, y-1), R (moving right) moves the player (x, y) -> (x+1, y), L (moving left) moves the player (x, y) -> (x-1, y). Always align your verbal directions (up/down/left/right) with these coordinate changes. When describing relative positions, use the coordinate conventions: "above" means larger y, "below" means smaller y, "right" means larger x, "left" means smaller x. Validate each step in the plan by explicitly computing the next (x, y) from the action; if the destination cell is a box, check the cell behind it and treat the move as a valid push if that cell is empty or a goal, otherwise the move is invalid and must not appear in the action sequence.
If the player pushes a box, the box will move one cell in the same direction as the player unless blocked by a wall or another box.
After any move (including a push), the player and any box must occupy different cells; they can never share the same location.

## Instructions
- Simulate each plan step-by-step using the Sokoban rules and the observation.
- Build each plan's step sequence as alternating entries: node -> action -> node -> action -> ...
- A node entry must include a node id and the predicted observation text.
- Node ids must be unique across all plans (no duplicate node_id between plans).
- IMPORTANT: Observations must include ONLY player location and box location (no walls, no goals, no box-on-goal).
- The observation text must start with a short "Thought: ..." line that explains how the previous action moves player/box while considering walls.
- An action entry must include only the action (U/D/L/R). No separate thought key in action entries.
- Preserve ALL actions from every plan (do not drop steps).
- JSON keys must appear in this order: plan_sequences.

Return JSON only:
{{{{
  "plan_sequences": [
    {{{{
      "plan_id": "plan1",
      "steps": [
        {{{{
          "node_id": "nodeX",
          "thought": "Initial states"
          "observation": "player location: (x, y)\nbox location: (x1, y1), ..."
        }}}},
        {{{{
          "action": "U"
        }}}},
        {{{{
          "node_id": "nodeY",
          "thought": "After moving up (U), player is at (x, y+1) unless blocked; box moves to (x1, y1+1) if pushed."
          "observation": "player location: (x, y+1)\nbox location: (x1, y1+1), ..."
        }}}}
      ]
    }}}}
  ]
}}}}
"""

GRAPH_STEP1_SYSTEM_CLAUDE = \
f"""Simulate Sokoban plans, produce per-plan step sequences and generate the graph.

{{sokoban_rule}}

IMPORTANT: Use the action-coordinate rules exactly. Actions update the player location: U (moving up) moves the player (x, y) -> (x, y+1), D (moving down) moves the player (x, y) -> (x, y-1), R (moving right) moves the player (x, y) -> (x+1, y), L (moving left) moves the player (x, y) -> (x-1, y). Always align your verbal directions (up/down/left/right) with these coordinate changes. When describing relative positions, use the coordinate conventions: "above" means larger y, "below" means smaller y, "right" means larger x, "left" means smaller x. Validate each step in the plan by explicitly computing the next (x, y) from the action; if the destination cell is a box, check the cell behind it and treat the move as a valid push if that cell is empty or a goal, otherwise the move is invalid and must not appear in the action sequence.
If the player pushes a box, the box will move one cell in the same direction as the player unless blocked by a wall or another box.
After any move (including a push), the player and any box must occupy different cells; they can never share the same location.

## Instructions
- Simulate each plan step-by-step using the Sokoban rules and the observation.
- Build each plan's step sequence as alternating entries: node -> action -> node -> action -> ...
- A node entry must include a node id and the predicted observation text.
- Node ids must be unique across all plans (no duplicate node_id between plans).
- IMPORTANT: Observations must include ONLY player location and box location (no walls, no goals, no box-on-goal).
- The observation text must start with a short "Thought: ..." line that explains how the previous action moves player/box while considering walls.
- An action entry must include only the action (U/D/L/R). No separate thought key in action entries.
- Preserve ALL actions from every plan (do not drop steps).
- JSON keys must appear in this order: plan_sequences.

Return JSON only:
{{{{
  "plan_sequences": [
    {{{{
      "plan_id": "plan1",
      "steps": [
        {{{{
          "kind": "node",
          "node_id": "nodeX",
          "thought": "Initial states"
          "observation": "player location: (x, y)\nbox location: (x1, y1), ..."
        }}}},
        {{{{
          "kind": "action",
          "action": "U"
        }}}},
        {{{{
          "kind": "node",
          "node_id": "nodeY",
          "thought": "After moving up (U), player is at (x, y+1) unless blocked; box moves to (x1, y1+1) if pushed."
          "observation": "player location: (x, y+1)\nbox location: (x1, y1+1), ..."
        }}}}
      ]
    }}}}
  ]
}}}}
"""

GRAPH_STEP2_INSTRUCTIONS = """## Instructions
- After all plan sequences are built, merge identical nodes when the observation text matches exactly.
- Preserve ALL actions from every plan (do not drop steps).
- Construct the full graph (nodes + edges) from the merged nodes.
- Generate a thought for each edge based on the transition between observations.
- Mark goal nodes explicitly with "is_goal": true in full_graph nodes.
- JSON keys must appear in this order: reasoning, merge_log, full_graph.
- In merge_log entries, put "reason" first, then "kept_node", then "merged_nodes".

Return JSON only:
{
  "reasoning": "overall reasoning for node merging and graph construction",
  "merge_log": [
    {
      "reason": "same observation text",
      "kept_node": "nodeX",
      "merged_nodes": ["nodeXX", ...]
    }
  ],
  "full_graph": {
    "nodes": [
      {
        "id": "nodeX",
        "observation": "player location: (x, y)\nbox location: (x1, y1), ...",
        "is_goal": true or false
      }
    ],
    "edges": [
      {
        "from": "nodeX",
        "to": "nodeY",
        "thought": "why this step is taken",
        "action": "<U/D/L/R>"
      }
    ]
  }
}
"""

PLAN_USER_TEMPLATE = \
f"""Observation:
{{observation}}
Steps remaining: {{steps_remaining}}
"""


GRAPH_STEP1_USER_TEMPLATE = \
f"""Observation:
{{observation}}
Steps remaining: {{steps_remaining}}
Plans text:
{{plans_text}}"""

GRAPH_STEP2_USER_TEMPLATE = \
f"""Use the plan_sequences you just produced.
{{instructions}}"""

PATH_SELECT_SYSTEM = \
"""You select a path from a directed graph.

Each node represents an observation; a node with is_goal true means all boxes are on goals.
Goal: choose a sequence of edges starting from the given start node, with length <= steps_remaining, that reaches a node with is_goal true.
Prefer a path that reaches any node with is_goal true within steps_remaining. If multiple exist, choose the shortest.
If none reach a goal, choose any valid path with the maximum length (<= steps_remaining).
Do not repeat nodes (avoid cycles).
Use only edges listed in full_graph.

Return JSON only (in each edge object, list keys in this order: from, to, thought, action):
{
  "edge_sequence": [
    {"from": "node1", "to": "node2", "thought": "why this step is taken", "action": "U"}
  ]
}
"""

PATH_SELECT_USER_TEMPLATE = """Start node id: {start_node_id}
Steps remaining: {steps_remaining}
Full graph JSON:
{full_graph_json}"""

SCORE_NODES_SYSTEM = \
f"""You score all the states in the graph. Higher score means closer to solving.
Goal: move all boxes onto goals (use goal locations from the observation).
Assign score 1.0 to goal states. Assign score -1.0 if, starting from this state, there is no valid sequence of actions that can ever reach any goal state (deadlocked/unreachable). All other states should have score 0.0.
Consider wall locations, blocked pushes, and required pushing routes when judging reachability.
Only use scores -1.0, 0.0, or 1.0 (no other values).
For each state, provide a short reasoning sentence before assigning its score.
Return JSON only with both reasons and scores, for example:
{{{{
  "reasons": {{{{
    "s0": "reasoning for s0: whether any sequence can reach a goal, considering walls, blocked pushes, and deadlock signals (e.g., box stuck in a corner with no goal)",
    ...
  }}}},g
  "scores": {{{{
    "s0": 0.0,
    ...
  }}}}
}}}}.
"""

SCORE_NODES_USER_TEMPLATE = """Observation:
{observation}
States:
{states}"""

SYSTEM_PROMPT_TEMPLATE = """
{{ INSTRUCTION }}

{% if examples_text %}
{{ icl_prompt }}
{{ examples_text }}
{% endif %}
"""

graph_step1_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "sokoban_plan_sequences",
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
                                        {
                                            "type": "object",
                                            "properties": {
                                                "action": {
                                                    "type": "string",
                                                    "enum": ["U", "D", "L", "R"],
                                                },
                                            },
                                            "required": ["action"],
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

graph_step1_response_format_claude = graph_step1_output_format = {
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
                                # step type discriminator
                                "kind": {"type": "string", "enum": ["node", "action"]},

                                # state fields
                                "node_id": {"type": "string"},
                                "thought": {"type": "string"},
                                "observation": {"type": "string"},

                                # action fields
                                "action": {"type": "string", "enum": ["U", "D", "L", "R"]},
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

graph_step2_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "sokoban_graph",
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
                                    "is_goal": {"type": "boolean"},
                                },
                                "required": ["id", "observation", "is_goal"],
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
                                    "action": {"type": "string", "enum": ["U", "D", "L", "R"]},
                                },
                                "required": ["from", "to", "thought", "action"],
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

graph_step2_response_format_claude = {
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
                            "is_goal": {"type": "boolean"},
                        },
                        "required": ["id", "observation", "is_goal"],
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
                            "action": {"type": "string", "enum": ["U", "D", "L", "R"]},
                        },
                        "required": ["from", "to", "thought", "action"],
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

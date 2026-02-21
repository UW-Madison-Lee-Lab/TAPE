"""
Graph-ILP Agent with All-Step Replanning.
Forces replanning at every step regardless of state match.
"""

from typing import Any, Dict, List, Optional, Tuple
from jinja2 import Template

from real_agents.base import BaseAgent
from utils.llm import client, pretty_print_conversation
from utils.llm import gemini_client
from utils.llm import anthropic_client
from env.sokoban import State
from utils.planning import ORTOOLS_AVAILABLE, select_path_ilp


SYSTEM_PROMPT = """Interact with Sokoban environment to solve a task (placing every box onto goals.)

## Sokoban rules (task + mechanics)
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
- Push blocked by box, D: player at (x, y), box at (x, y - 1), another box at (x, y - 2). Action D -> invalid, player stays at (x, y), boxes stay at (x, y - 1) and (x, y - 2); box on goal location unchanged.

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
Action: <U|D|L|R>"""


PLAN_PROMPT = """You are the player in a Sokoban environment. Generate a plan to place every box on a goal within a limited number of actions.

## Sokoban rules (task + mechanics)
- Task objective: place every box onto goals. The puzzle is solved when all boxes are on goals.
- Action mechanics: each action moves the player exactly one cell in the chosen direction (U/D/L/R).
    - U: move up, e.g., player at (x, y) -> player at (x, y + 1)
    - D: move down, e.g., player at (x, y) -> player at (x, y - 1)
    - L: move left, e.g., player at (x, y) -> player at (x - 1, y)
    - R: move right, e.g., player at (x, y) -> player at (x + 1, y)

## Current Observation
{{ observation }}

## Instruction
Generate a complete plan to solve this puzzle within {{ steps_remaining }} steps.
Output the plan as a sequence of actions (U/D/L/R) separated by commas.
Example: U, U, L, D, R

Plan:"""


def parse_plan(plan_text: str) -> List[str]:
    """Parse a plan string into a list of actions."""
    actions = []
    for part in plan_text.replace("\n", ",").split(","):
        part = part.strip().upper()
        if part in ("U", "D", "L", "R"):
            actions.append(part)
    return actions


def parse_action(text: str) -> Optional[str]:
    """Extract the action from a response."""
    import re
    match = re.search(r"Action:\s*([UDLR])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # fallback: look for standalone action
    for line in reversed(text.strip().split("\n")):
        line = line.strip().upper()
        if line in ("U", "D", "L", "R"):
            return line
    return None


class OursGraphILPAgentAllStepReplanning(BaseAgent):
    """
    Graph-ILP Agent that forces replanning at every step.
    This is equivalent to the threading bug behavior but done intentionally.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        fine_tuned_model: Optional[str] = None,
        temperature: float = 0.3,
        num_plans: int = 4,
        is_print: bool = False,
        repetition_penalty: float = 1.0,
    ):
        super().__init__(model=model, temperature=temperature)
        self.fine_tuned_model = fine_tuned_model
        self.num_plans = num_plans
        self.is_print = is_print
        self.repetition_penalty = repetition_penalty
        self.system_prompt = SYSTEM_PROMPT

        # Replanning tracking
        self.replanning_log: List[Dict[str, Any]] = []
        
        # Graph log
        self.graph_log_base: Optional[str] = None
        self.graph_log_index: int = 0

    def reset_episode(self) -> None:
        """Reset episode-specific state."""
        self.replanning_log = []
        self.graph_log_index = 0

    def set_graph_log_base(self, base: str) -> None:
        """Set the base path for graph logs."""
        self.graph_log_base = base
        self.graph_log_index = 0
        self.replanning_log = []

    def get_replanning_info(self) -> Dict[str, Any]:
        """Return replanning statistics for logging."""
        return {
            "num_replanning": len(self.replanning_log),
            "replanning_states": self.replanning_log,
        }

    def generate_plan(self, observation: str, steps_remaining: int) -> str:
        """Generate initial plan (not used in all-step replanning, but kept for interface)."""
        return ""

    def _parse_observation(self, observation: str) -> State:
        """Parse observation string into State object."""
        import re
        
        # Parse player location
        player_match = re.search(r"player location:\s*\((\d+),\s*(\d+)\)", observation)
        if not player_match:
            raise ValueError("Could not parse player location")
        player = (int(player_match.group(1)), int(player_match.group(2)))
        
        # Parse box locations
        boxes = []
        box_match = re.search(r"box location:\s*([^\n]+)", observation)
        if box_match:
            box_str = box_match.group(1)
            if "none" not in box_str.lower():
                for m in re.finditer(r"\((\d+),\s*(\d+)\)", box_str):
                    boxes.append((int(m.group(1)), int(m.group(2))))
        
        # Parse box on goal locations
        box_on_goal_match = re.search(r"box on goal location:\s*([^\n]+)", observation)
        if box_on_goal_match:
            bog_str = box_on_goal_match.group(1)
            if "none" not in bog_str.lower():
                for m in re.finditer(r"\((\d+),\s*(\d+)\)", bog_str):
                    boxes.append((int(m.group(1)), int(m.group(2))))
        
        # Parse walls
        walls = set()
        wall_match = re.search(r"wall location:\s*([^\n]+)", observation)
        if wall_match:
            wall_str = wall_match.group(1)
            for m in re.finditer(r"\((\d+),\s*(\d+)\)", wall_str):
                walls.add((int(m.group(1)), int(m.group(2))))
        
        # Parse goals
        goals = set()
        goal_match = re.search(r"goal location:\s*([^\n]+)", observation)
        if goal_match:
            goal_str = goal_match.group(1)
            for m in re.finditer(r"\((\d+),\s*(\d+)\)", goal_str):
                goals.add((int(m.group(1)), int(m.group(2))))
        
        # Parse steps remaining
        steps_match = re.search(r"Steps remaining:\s*(\d+)", observation)
        steps_remaining = int(steps_match.group(1)) if steps_match else 0
        
        return State(
            player=player,
            boxes=frozenset(boxes),
            walls=walls,
            goals=goals,
        ), steps_remaining

    def _generate_diverse_plans(
        self, observation: str, steps_remaining: int
    ) -> List[Tuple[List[str], List[State]]]:
        """Generate multiple diverse plans."""
        from env.sokoban import SokobanEnv, State
        
        current_state, _ = self._parse_observation(observation)
        
        # Create a temporary environment to simulate plans
        plans_with_states = []
        
        for plan_idx in range(self.num_plans):
            # Generate a plan using LLM
            plan_prompt = Template(PLAN_PROMPT).render(
                observation=observation,
                steps_remaining=steps_remaining,
            )
            
            messages = [{"role": "user", "content": plan_prompt}]
            
            if "gemini" in self.model:
                response = gemini_client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": [{"text": plan_prompt}]}],
                    config={
                        "temperature": self.temperature + 0.1 * plan_idx,  # Increase diversity
                        "max_output_tokens": 256,
                    },
                )
                plan_text = response.text
            elif "claude" in self.model:
                response = anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=min(1.0, self.temperature + 0.1 * plan_idx),
                    messages=messages,
                )
                plan_text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.fine_tuned_model or self.model,
                    messages=messages,
                    temperature=self.temperature + 0.1 * plan_idx,
                    max_tokens=256,
                )
                plan_text = response.choices[0].message.content
            
            actions = parse_plan(plan_text)
            if actions:
                # Simulate the plan to get state sequence
                state_sequence = [current_state]
                # For simplicity, just store actions without full simulation
                plans_with_states.append((actions, state_sequence))
        
        return plans_with_states

    def act(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Execute one step with all-step replanning.
        Always generates a new plan at each step.
        """
        # Extract current observation from the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if not last_user_msg:
            return {"role": "assistant", "content": "Thought: No observation found.\nAction: U"}
        
        # Parse current state
        try:
            current_state, steps_remaining = self._parse_observation(last_user_msg)
        except Exception as e:
            return {"role": "assistant", "content": f"Thought: Failed to parse observation: {e}\nAction: U"}
        
        # Record replanning event
        def state_to_dict_for_log(st: State) -> Dict[str, Any]:
            return {
                "player": [st.player[0], st.player[1]],
                "boxes": [[x, y] for x, y in st.boxes],
            }
        self.replanning_log.append({
            "state": state_to_dict_for_log(current_state),
            "steps_remaining": steps_remaining,
        })
        
        # Generate diverse plans
        plans = self._generate_diverse_plans(last_user_msg, steps_remaining)
        
        if not plans:
            return {"role": "assistant", "content": "Thought: Could not generate any plans.\nAction: U"}
        
        # Select the best plan using ILP if available
        if ORTOOLS_AVAILABLE and len(plans) > 1:
            try:
                selected_idx = select_path_ilp(
                    [p[0] for p in plans],  # Just actions
                    steps_remaining,
                )
                chosen_actions = plans[selected_idx][0]
            except Exception:
                chosen_actions = plans[0][0]
        else:
            chosen_actions = plans[0][0]
        
        if not chosen_actions:
            return {"role": "assistant", "content": "Thought: Empty plan generated.\nAction: U"}
        
        # Take the first action from the plan
        next_action = chosen_actions[0]
        
        # Generate hint for strong conditioning
        hint = self._generate_hint(current_state, next_action)
        
        # Build response
        thought = f"Replanning at step {steps_remaining}. Generated plan: {', '.join(chosen_actions[:5])}{'...' if len(chosen_actions) > 5 else ''}. Taking first action."
        
        response_content = f"Thought: {thought}\nAction: {next_action}"
        
        if self.is_print:
            print(f"[AllStepReplanning] Step {steps_remaining}: {next_action} (from plan of {len(chosen_actions)} actions)")
        
        self.graph_log_index += 1
        
        return {"role": "assistant", "content": response_content}

    def _generate_hint(self, state: State, action: str) -> str:
        """Generate a hint describing the action effect."""
        player = state.player
        dx, dy = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}[action]
        new_pos = (player[0] + dx, player[1] + dy)
        
        if new_pos in state.boxes:
            box_new = (new_pos[0] + dx, new_pos[1] + dy)
            return f"Player pushes box from {new_pos} to {box_new}; player moves to {new_pos}."
        else:
            return f"Player moves {action} from {player} to {new_pos} with no obstruction."
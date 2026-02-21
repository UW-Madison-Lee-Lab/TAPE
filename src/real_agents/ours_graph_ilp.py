import os
import json
import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from jinja2 import Template

from real_agents.base import BaseAgent
from utils.llm import client, pretty_print_conversation
from utils.llm import gemini_client
from utils.llm import anthropic_client
from env.sokoban import State
from env.alfworld import StateAlfworld
from utils.planning import ORTOOLS_AVAILABLE, select_path_ilp
from real_agents.prompts.ours_graph_ilp_sokoban_prompts import (
    RULE as SOKOBAN_RULE,
    REACT_INSTRUCTION as SOKOBAN_REACT_INSTRUCTION,
    PLAN_INSTRUCTION as SOKOBAN_PLAN_INSTRUCTION,
    GRAPH_STEP1_SYSTEM as SOKOBAN_GRAPH_STEP1_SYSTEM,
    GRAPH_STEP1_SYSTEM_CLAUDE as SOKOBAN_GRAPH_STEP1_SYSTEM_CLAUDE,
    GRAPH_STEP2_INSTRUCTIONS as SOKOBAN_GRAPH_STEP2_INSTRUCTIONS,
    PLAN_USER_TEMPLATE as SOKOBAN_PLAN_USER_TEMPLATE,
    GRAPH_STEP1_USER_TEMPLATE as SOKOBAN_GRAPH_STEP1_USER_TEMPLATE,
    GRAPH_STEP2_USER_TEMPLATE as SOKOBAN_GRAPH_STEP2_USER_TEMPLATE,
    PATH_SELECT_SYSTEM as SOKOBAN_PATH_SELECT_SYSTEM,
    PATH_SELECT_USER_TEMPLATE as SOKOBAN_PATH_SELECT_USER_TEMPLATE,
    SCORE_NODES_SYSTEM as SOKOBAN_SCORE_NODES_SYSTEM,
    SCORE_NODES_USER_TEMPLATE as SOKOBAN_SCORE_NODES_USER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE as SOKOBAN_SYSTEM_PROMPT_TEMPLATE,
    graph_step1_response_format as sokoban_graph_step1_response_format,
    graph_step1_response_format_claude as sokoban_graph_step1_response_format_claude,
    graph_step2_response_format as sokoban_graph_step2_response_format,
    graph_step2_response_format_claude as sokoban_graph_step2_response_format_claude,
)
from real_agents.prompts.ours_graph_ilp_alfworld_prompts import (
    REACT_INSTRUCTION_ALFWORLD as ALFWORLD_REACT_INSTRUCTION,
    PLAN_INSTRUCTION as ALFWORLD_PLAN_INSTRUCTION,
    GRAPH_STEP1_SYSTEM as ALFWORLD_GRAPH_STEP1_SYSTEM,
    GRAPH_STEP2_INSTRUCTIONS as ALFWORLD_GRAPH_STEP2_INSTRUCTIONS,
    PLAN_USER_TEMPLATE as ALFWORLD_PLAN_USER_TEMPLATE,
    GRAPH_STEP1_USER_TEMPLATE as ALFWORLD_GRAPH_STEP1_USER_TEMPLATE,
    GRAPH_STEP2_USER_TEMPLATE as ALFWORLD_GRAPH_STEP2_USER_TEMPLATE,
    SCORE_NODES_SYSTEM as ALFWORLD_SCORE_NODES_SYSTEM,
    SCORE_NODES_USER_TEMPLATE as ALFWORLD_SCORE_NODES_USER_TEMPLATE,
    OBSERVATION_EXTRACT_SYSTEM as ALFWORLD_OBSERVATION_EXTRACT_SYSTEM,
    OBSERVATION_EXTRACT_USER_TEMPLATE as ALFWORLD_OBSERVATION_EXTRACT_USER_TEMPLATE,
    STATE_EQUAL_SYSTEM as ALFWORLD_STATE_EQUAL_SYSTEM,
    STATE_EQUAL_USER_TEMPLATE as ALFWORLD_STATE_EQUAL_USER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE as ALFWORLD_SYSTEM_PROMPT_TEMPLATE,
)
from real_agents.prompts.ours_graph_ilp_arithmetic_prompts import (
    PLAN_INSTRUCTION_TEMPLATE as ARITH_PLAN_INSTRUCTION_TEMPLATE,
    GRAPH_STEP1_SYSTEM_TEMPLATE as ARITH_GRAPH_STEP1_SYSTEM_TEMPLATE,
    GRAPH_STEP2_INSTRUCTIONS as ARITH_GRAPH_STEP2_INSTRUCTIONS,
    PLAN_USER_TEMPLATE as ARITH_PLAN_USER_TEMPLATE,
    GRAPH_STEP1_USER_TEMPLATE as ARITH_GRAPH_STEP1_USER_TEMPLATE,
    GRAPH_STEP2_USER_TEMPLATE as ARITH_GRAPH_STEP2_USER_TEMPLATE,
    SCORE_NODES_SYSTEM as ARITH_SCORE_NODES_SYSTEM,
    SCORE_NODES_USER_TEMPLATE as ARITH_SCORE_NODES_USER_TEMPLATE,
    REACT_INSTRUCTION_TEMPLATE as ARITH_REACT_INSTRUCTION_TEMPLATE,
    _collect_tool_names as arith_collect_tool_names,
    _tool_description_block as arith_tool_description_block,
    build_graph_step1_response_format as arith_build_graph_step1_response_format,
    build_graph_step1_response_format_claude as arith_build_graph_step1_response_format_claude,
    build_graph_step2_response_format as arith_build_graph_step2_response_format,
    build_graph_step2_response_format_claude as arith_build_graph_step2_response_format_claude,
)
import random


GraphState = Union[StateAlfworld]

class OursGraphILPAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        fine_tuned_model: Optional[str] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        debug_mode: bool = False,
        instruct_tuned: bool = True,
        enable_lora: bool = False,
        lora=None,
        is_print: bool = False,
        stop: Optional[List[str]] = None,
        method: str = "react",
        num_plans: int = 4,
        use_solver: bool = True,
        use_replanning: bool = True,
        use_strong_conditioning: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            fine_tuned_model=fine_tuned_model,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            debug_mode=debug_mode,
            instruct_tuned=instruct_tuned,
            enable_lora=enable_lora,
            lora=lora,
            is_print=is_print,
        )
        if "gemini" in model:
            self.client = gemini_client
        elif "claude" in model:
            self.client = anthropic_client
        else:
            self.client = client
        self.model = model
        self.temperature = temperature
        self.stop = stop or None
        # Predeclare attributes set later to satisfy linters/type checkers
        self.system_prompt = None
        self.user_prompt = None
        self.method = method
        self.plan_text: Optional[str] = None
        self.cached_actions: List[str] = []
        self.cached_state_sequence: Optional[List[State]] = None
        self.cached_thoughts: Optional[List[str]] = None
        self.cached_action_index = 0
        self.graph_log_base: Optional[str] = None
        self.graph_log_index = 0
        self.num_plans = num_plans
        self.use_solver = use_solver
        self.use_replanning = use_replanning
        self.use_strong_conditioning = use_strong_conditioning
        # Replanning tracking: list of (state_dict, steps_remaining) at each replanning
        self.replanning_log: List[Dict[str, Any]] = []

    def reset_episode(self) -> None:
        self.plan_text = None
        self.is_plan = False
        self.cached_actions = []
        self.cached_state_sequence = None
        self.cached_thoughts = None
        self.cached_action_index = 0
        self.graph_log_index = 0
        self.replanning_log = []

    def set_graph_log_base(self, base_path: Optional[str]) -> None:
        self.graph_log_base = base_path
        self.graph_log_index = 0
        self.replanning_log = []
    
    def get_replanning_info(self) -> Dict[str, Any]:
        """Return replanning statistics and log for external logging."""
        return {
            "num_replanning": len(self.replanning_log),
            "replanning_states": self.replanning_log,
        }

    def get_examples_for_task(self, task_type: str) -> tuple[str, int]:
        """Load ICL examples and concatenate their text with newlines.

        Returns a tuple of (examples_text, num_examples). If none found, returns ("", 0).
        """
        base_dir = os.path.dirname(__file__)
        if self.method == "react":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl.json")
        elif self.method == "react_preplan":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl.json")
        elif self.method == "react_planning":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl_replanning_v2.json")
        elif self.method == "react_formal_planning":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl_replanning_v2.json")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        with open(icl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodes = data.get(task_type, []) or []
        # Each episode is a list of {role, content}; flatten each to lines of content
        episode_texts: List[str] = []
        for ep in episodes:
            contents = [m.get("content", "") for m in (ep or [])]
            # Join messages in an episode with newlines
            episode_texts.append("\n".join(filter(None, contents)))
        assert len(episode_texts) > 0, f"No examples found for task type: {task_type}"
        return episode_texts, len(episode_texts)

    def _is_fixed_temp_model(self, model_name: str) -> bool:
        """Models like o1, o3, and gpt-5 families don't accept temperature.

        Returns True if temperature should be omitted for this model.
        """
        name = (model_name or "").lower()
        return (
            name.startswith("o1")
            or name.startswith("o3")
            or name.startswith("gpt-5")
            or name.startswith("gpt5")
        )

    def _extract_observation(self, text: str) -> Tuple[str, int]:
        steps_match = re.search(r"Steps remaining:\s*(\d+)", text, re.IGNORECASE)
        steps = int(steps_match.group(1)) if steps_match else 0
        if "Observation:" in text:
            obs_part = text.split("Observation:", 1)[1]
            obs_part = obs_part.split("Steps remaining:", 1)[0].strip()
            return obs_part, steps
        return text.strip(), steps

    def generate_plan(self, observation: str, steps_remaining: int) -> str:
        return None

    def act(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        def state_to_dict(st: State) -> Dict[str, Any]:
            return {
                "player": [st.player[0], st.player[1]],
                "boxes": [[x, y] for x, y in st.boxes],
            }

        def write_graph_log(
            *,
            full_graph: Dict[str, Any],
            merge_log: List[Dict[str, Any]],
            states: List[State],
            scores: Dict[State, float],
            actions: Optional[List[str]],
            state_sequence: Optional[List[State]],
            steps_remaining: int,
        ) -> None:
            if not self.graph_log_base:
                return
            self.graph_log_index += 1
            out_path = f"{self.graph_log_base}_graph{self.graph_log_index}.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            score_list = [
                {"state": state_to_dict(st), "score": scores.get(st, 0.0)}
                for st in sorted(states, key=lambda s: (s.player[1], s.player[0], s.boxes))
            ]
            payload: Dict[str, Any] = {
                "steps_remaining": steps_remaining,
                "graph": full_graph,
                "merge_log": merge_log,
                "node_scores": score_list,
                "actions": actions or [],
            }
            if state_sequence:
                payload["state_sequence"] = [state_to_dict(st) for st in state_sequence]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
        def chat_with_print(
            prompt_messages: List[Dict],
            response_format: Optional[Dict[str, Any]] = None,
            max_tokens: Optional[int] = None,
        ):
            response = self._chat(
                messages=prompt_messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            if "claude" in self.model:
                content = response.content[0].text
                role = "assistant"
            else:
                role = response.choices[0].message.role
                content = response.choices[0].message.content

            if self.is_print:
                pretty_print_conversation(
                    prompt_messages
                    + [
                        {
                            "role": role,
                            "content": content,
                        }
                    ]
                )
            return response

        def extract_json(text: str) -> Optional[str]:
            fenced = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, re.DOTALL | re.IGNORECASE)
            if fenced:
                return fenced.group(1)
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            return text[start : end + 1]

        def parse_coords(raw: str) -> List[Tuple[int, int]]:
            if raw is None:
                return []
            text = str(raw)
            if not text or "none" in text.lower():
                return []
            coords = []
            matches = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
            if matches:
                for x_str, y_str in matches:
                    coords.append((int(x_str), int(y_str)))
                return coords
            numbers = re.findall(r"-?\\d+", text)
            for i in range(0, len(numbers) - 1, 2):
                coords.append((int(numbers[i]), int(numbers[i + 1])))
            return coords

        def parse_observation(obs_text: str) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...], set, set]:
            lines = [ln.strip() for ln in obs_text.splitlines() if ln.strip()]
            data: Dict[str, str] = {}
            for line in lines:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                data[key.strip().lower()] = val.strip()
            walls = set(parse_coords(data.get("wall location", "")))
            goals = set(parse_coords(data.get("goal location", "")))
            player_list = parse_coords(data.get("player location", ""))
            player = player_list[0] if player_list else (0, 0)
            boxes = parse_coords(data.get("box location", "")) + parse_coords(
                data.get("box on goal location", "")
            )
            return player, tuple(sorted(boxes)), walls, goals

        def parse_graph_observation(obs_text: str) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
            text = obs_text.replace("\\n", "\n")

            def extract_value_after_label(label_pattern: str) -> str:
                match = re.search(rf"{label_pattern}\s*:\s*", text, re.IGNORECASE)
                if not match:
                    return ""
                rest = text[match.end():]
                next_label = re.search(
                    r"\n\s*(?:\w+\s+location\s*:|step remaining\s*:)",
                    rest,
                    re.IGNORECASE,
                )
                if next_label:
                    rest = rest[: next_label.start()]
                inline_label = re.search(r"\b\w+\s+location\s*:", rest, re.IGNORECASE)
                if inline_label:
                    rest = rest[: inline_label.start()]
                return rest.strip()

            player_raw = extract_value_after_label(r"player location")
            box_raw = extract_value_after_label(r"box locations?")
            box_on_goal_raw = extract_value_after_label(r"box on goal locations?")
            player_list = parse_coords(player_raw)
            player = player_list[0] if player_list else (0, 0)
            boxes = parse_coords(box_raw) + parse_coords(box_on_goal_raw)
            return player, tuple(sorted(boxes))

        def apply_action(
            st: State,
            action: str,
            walls: set,
        ) -> Optional[State]:
            deltas = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}
            if action not in deltas:
                return None
            action = action.upper()
            dx, dy = deltas[action]
            px, py = st.player
            nx, ny = px + dx, py + dy
            if (nx, ny) in walls:
                return None
            boxes = set(st.boxes)
            if (nx, ny) in boxes:
                bx, by = nx + dx, ny + dy
                if (bx, by) in walls or (bx, by) in boxes:
                    return None
                boxes.remove((nx, ny))
                boxes.add((bx, by))
                return State((nx, ny), tuple(sorted(boxes)))
            return State((nx, ny), st.boxes)

        def is_goal_state(st: State, goals: set) -> bool:
            return set(st.boxes) == set(goals)

        def fallback_action_from_plans() -> str:
            return random.choice(["U", "D", "L", "R"])

        def format_solver_hint(thought: Optional[str], action: Optional[str]) -> str:
            thought_text = (thought or "").strip()
            if thought_text:
                return thought_text
            if action in {"U", "D", "L", "R"}:
                return f"Suggested action: {action}"
            return ""

        def append_hint_to_messages(target_messages: List[Dict], hint_text: str) -> None:
            if not hint_text:
                return
            hint_line = f"\nHint: {hint_text}"
            if target_messages and target_messages[-1].get("role") == "user":
                target_messages[-1]["content"] = f"{target_messages[-1]['content']}{hint_line}"
                return
            target_messages.append({"role": "user", "content": f"Hint: {hint_text}"})

        def respond_with_hint(target_messages: List[Dict], hint_text: str) -> List[str]:
            append_hint_to_messages(target_messages, hint_text)
            response = self._chat(
                messages=target_messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
            target_messages.append(
                {
                    "content": response.choices[0].message.content,
                    "role": response.choices[0].message.role,
                }
            )
            self.logger(target_messages)
            return [choice.message.content for choice in response.choices]

        def render_action_response(action: Optional[str], thought: str) -> List[str]:
            next_action = f"Thought: {thought}\nAction: {action}"
            if self.use_strong_conditioning:
                hint_text = format_solver_hint(thought, action)
                return respond_with_hint(messages, hint_text)
            messages.append({"content": next_action, "role": "assistant"})
            self.logger(messages)
            return [next_action]

        def select_path_llm(
            *,
            full_graph: Dict[str, Any],
            start_node_id: str,
            steps_remaining: int,
        ) -> List[Dict[str, str]]:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sokoban_path_selection",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "edge_sequence": {
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
                            },
                        },
                        "required": ["edge_sequence"],
                        "additionalProperties": False,
                    },
                },
            }
            user_prompt = SOKOBAN_PATH_SELECT_USER_TEMPLATE.format(
                start_node_id=start_node_id,
                steps_remaining=steps_remaining,
                full_graph_json=json.dumps(full_graph, ensure_ascii=True),
            )
            response = chat_with_print(
                [
                    {"role": "system", "content": SOKOBAN_PATH_SELECT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
            )
            if "claude" in self.model:
                text = response.content[0].text
            else:
                text = response.choices[0].message.content
            data = None
            json_blob = extract_json(text)
            if json_blob:
                try:
                    data = json.loads(json_blob)
                except json.JSONDecodeError:
                    data = None
            if not isinstance(data, dict):
                return []
            edges = data.get("edge_sequence", [])
            return edges if isinstance(edges, list) else []

        def score_nodes_llm(
            states: List[State],
            observation_text: str,
            budget: int,
        ) -> Dict[State, float]:
            if not states:
                return {}
            state_ids = {st: f"node{i}" for i, st in enumerate(states)}
            lines = []
            for st in states:
                sid = state_ids[st]
                boxes = ", ".join(f"({x}, {y})" for x, y in st.boxes) if st.boxes else "none"
                lines.append(f"{sid}: player=({st.player[0]}, {st.player[1]}), boxes=[{boxes}]")
            system_prompt = SOKOBAN_SCORE_NODES_SYSTEM
            user_prompt = SOKOBAN_SCORE_NODES_USER_TEMPLATE.format(
                observation=observation_text,
                budget=budget,
                states="\n".join(lines),
            )
            response = chat_with_print(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if "claude" in self.model:
                text = response.content[0].text
            else:
                text = response.choices[0].message.content
            
            data = None
            json_blob = extract_json(text)
            if json_blob:
                try:
                    data = json.loads(json_blob)
                except json.JSONDecodeError:
                    data = None
            scores_raw = data.get("scores", data) if isinstance(data, dict) else None
            scores: Dict[State, float] = {}
            if isinstance(scores_raw, dict):
                for key, val in scores_raw.items():
                    st = next((s for s, sid in state_ids.items() if sid == key), None)
                    if st is None:
                        continue
                    try:
                        score = float(val)
                    except (TypeError, ValueError):
                        continue
                    if score <= -0.5:
                        scores[st] = -1.0
                    elif score >= 0.5:
                        scores[st] = 1.0
                    else:
                        scores[st] = 0.0
            return scores

        last_msg = messages[-1]["content"] if messages else ""
        obs_text, steps_remaining = self._extract_observation(last_msg)
        if steps_remaining <= 0:
            next_action = f"Thought: No steps remaining, taking a default move.\nAction: {random.choice(['U', 'D', 'L', 'R'])}"
            messages.append({"content": next_action, "role": "assistant"})
            self.logger(messages)
            return [next_action]

        graph_player, graph_boxes = parse_graph_observation(obs_text)
        current_graph_state = State(graph_player, graph_boxes)
        if self.cached_actions:
            if self.use_replanning and self.cached_state_sequence:
                match_index = None
                start_idx = min(self.cached_action_index, len(self.cached_state_sequence) - 1)
                if start_idx + 1 < len(self.cached_state_sequence):
                    if self.cached_state_sequence[start_idx + 1] == current_graph_state:
                        match_index = start_idx + 1
                else:
                    match_index = None

                if match_index is not None and match_index < len(self.cached_actions):
                    chosen_action = self.cached_actions[match_index]
                    cached_thought = None
                    if self.cached_thoughts and match_index < len(self.cached_thoughts):
                        cached_thought = self.cached_thoughts[match_index]
                    self.cached_action_index = match_index + 1
                    if chosen_action not in {"U", "D", "L", "R"}:
                        chosen_action = None
                    thought_text = cached_thought or "Continuing cached path."
                    return render_action_response(chosen_action, thought_text)
                if match_index is None:
                    self.cached_actions = []
                    self.cached_state_sequence = None
                    self.cached_thoughts = None
                    self.cached_action_index = 0
            else:
                if self.cached_action_index < len(self.cached_actions):
                    chosen_action = self.cached_actions[self.cached_action_index]
                    cached_thought = None
                    if self.cached_thoughts and self.cached_action_index < len(self.cached_thoughts):
                        cached_thought = self.cached_thoughts[self.cached_action_index]
                    self.cached_action_index += 1
                    if chosen_action not in {"U", "D", "L", "R"}:
                        chosen_action = None
                    thought_text = cached_thought or "Continuing cached path."
                    return render_action_response(chosen_action, thought_text)
                self.cached_actions = []
                self.cached_state_sequence = None
                self.cached_thoughts = None
                self.cached_action_index = 0

        player, boxes, walls, goals = parse_observation(obs_text)
        start_state = State(player, boxes)

        # Record replanning event with current state
        def state_to_dict_for_log(st: State) -> Dict[str, Any]:
            return {
                "player": [st.player[0], st.player[1]],
                "boxes": [[x, y] for x, y in st.boxes],
            }
        self.replanning_log.append({
            "state": state_to_dict_for_log(start_state),
            "steps_remaining": steps_remaining,
        })

        # 1) M plan path sampling. One output should contain M diverse plans.
        plan_system = SOKOBAN_PLAN_INSTRUCTION.format(
            sokoban_rule=SOKOBAN_RULE,
            num_plans=self.num_plans,
        )

        plan_user = SOKOBAN_PLAN_USER_TEMPLATE.format(
            observation=obs_text,
            steps_remaining=steps_remaining,
        )

        plan_response = chat_with_print(
            [
                {"role": "system", "content": plan_system},
                {"role": "user", "content": plan_user},
            ],
        )
        if "claude" in self.model:
            plan_text = plan_response.content[0].text
        else:
            plan_text = plan_response.choices[0].message.content

        # 2) Step 1: simulate plans and output per-plan observation sequences.
        if "claude" in self.model:
            graph_step1_system = SOKOBAN_GRAPH_STEP1_SYSTEM_CLAUDE.format(
                sokoban_rule=SOKOBAN_RULE,
            )
        else:
            graph_step1_system = SOKOBAN_GRAPH_STEP1_SYSTEM.format(
                sokoban_rule=SOKOBAN_RULE,
            )
        graph_step1_user = SOKOBAN_GRAPH_STEP1_USER_TEMPLATE.format(
            observation=obs_text,
            steps_remaining=steps_remaining,
            plans_text=plan_text,
        )
        graph_messages = [
            {"role": "system", "content": graph_step1_system},
            {"role": "user", "content": graph_step1_user},
        ]

        graph_step1_response = self._chat(
            messages=graph_messages,
            response_format=sokoban_graph_step1_response_format_claude if "claude" in self.model else sokoban_graph_step1_response_format,
        )
        if "claude" in self.model:
            graph_step1_text = graph_step1_response.content[0].text
            graph_step1_role = "assistant"
        else:
            graph_step1_text = graph_step1_response.choices[0].message.content
            graph_step1_role = graph_step1_response.choices[0].message.role
        graph_messages.append(
            {
                "role": graph_step1_role,
                "content": graph_step1_text,
            }
        )
        if self.is_print:
            pretty_print_conversation(graph_messages)

        # 3) Step 2: merge observations and build full graph.
        graph_step2_user = SOKOBAN_GRAPH_STEP2_USER_TEMPLATE.format(
            instructions=SOKOBAN_GRAPH_STEP2_INSTRUCTIONS,
        )

        graph_messages.append({"role": "user", "content": graph_step2_user})
        graph_step2_response = self._chat(
            messages=graph_messages,
            response_format=sokoban_graph_step2_response_format_claude if "claude" in self.model else sokoban_graph_step2_response_format,
        )

        if "claude" in self.model:
            graph_step2_text = graph_step2_response.content[0].text
            graph_step2_role = "assistant"
        else:
            graph_step2_text = graph_step2_response.choices[0].message.content
            graph_step2_role = graph_step2_response.choices[0].message.role
        graph_messages.append(
            {
                "role": graph_step2_role,
                "content": graph_step2_text,
            }
        )
        if self.is_print:
            pretty_print_conversation(graph_messages[-2:])
        graph_step2_data: Optional[Dict[str, Any]] = None
        try:
            graph_step2_data = json.loads(graph_step2_text)
        except json.JSONDecodeError:
            graph_blob = extract_json(graph_step2_text)
            if graph_blob:
                graph_step2_data = json.loads(graph_blob)
        if not isinstance(graph_step2_data, dict):
            raise ValueError("Step 2 response must be a JSON object.")
        merge_log_data = graph_step2_data.get("merge_log")
        full_graph_data = graph_step2_data.get("full_graph")
        if not isinstance(merge_log_data, list):
            raise ValueError("Step 2 response missing merge_log.")
        if not isinstance(full_graph_data, dict):
            raise ValueError("Step 2 response missing full_graph.")
        nodes_data = full_graph_data.get("nodes")
        edges_data = full_graph_data.get("edges")
        if not isinstance(nodes_data, list) or not isinstance(edges_data, list):
            raise ValueError("Step 2 response missing nodes/edges in full_graph.")

        id_to_state: Dict[str, State] = {}
        obs_to_state: Dict[str, State] = {}
        goal_state_ids: List[str] = []
        for node in nodes_data:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            observation = node.get("observation")
            is_goal = node.get("is_goal", False)
            if not isinstance(node_id, str) or not isinstance(observation, str):
                continue
            player_xy, box_tuple = parse_graph_observation(observation)
            st = State(player_xy, box_tuple)
            id_to_state[node_id] = st
            obs_to_state[node_id] = st
            if isinstance(is_goal, bool) and is_goal:
                goal_state_ids.append(node_id)

        V = set(obs_to_state.values())
        E: List[Tuple[State, str, State]] = []
        edge_thoughts: Dict[Tuple[State, str, State], str] = {}
        for edge in edges_data:
            if not isinstance(edge, dict):
                continue
            from_id = edge.get("from")
            to_id = edge.get("to")
            thought = edge.get("thought", "")
            action_token = edge.get("action")
            if (
                not isinstance(from_id, str)
                or not isinstance(to_id, str)
                or not isinstance(action_token, str)
                or action_token.upper() not in {"U", "D", "L", "R"}
            ):
                continue
            u = id_to_state.get(from_id)
            v = id_to_state.get(to_id)
            if u is None or v is None:
                continue
            action_token = action_token.upper()
            key = (u, action_token, v)
            E.append(key)
            edge_thoughts[key] = str(thought) if thought is not None else ""
        if start_state not in V:
            V.add(start_state)

        fallback_action = fallback_action_from_plans()
        if not V or not E:
            self.cached_actions = []
            self.cached_state_sequence = None
            self.cached_thoughts = None
            self.cached_action_index = 0
            next_action = (
                "Thought: There is no solution. Select a random action.\n"
                f"Action: {fallback_action}"
            )
            messages.append({"content": next_action, "role": "assistant"})
            self.logger(messages)
            return [next_action]

        state_list = sorted(V, key=lambda s: (s.player[1], s.player[0], s.boxes))
        node_score: Dict[State, float] = {}
        actions = None
        state_sequence = None
        selection_thoughts: List[str] = []
        if self.use_solver:
            # 3) Graph scoring & constraint prediction (LLM estimates node scores).
            node_score = score_nodes_llm(state_list, obs_text, steps_remaining)
            if not node_score:
                node_score = {}

            # 4) Run ILP to solve the problem.
            goals_states = {id_to_state[node_id] for node_id in goal_state_ids if node_id in id_to_state}
            out_degree: Dict[State, int] = {st: 0 for st in V}
            for u, _a, _v in E:
                out_degree[u] = out_degree.get(u, 0) + 1
            for st in goals_states:
                node_score[st] = 1.0
            for st in V:
                if st not in goals_states and out_degree.get(st, 0) == 0:
                    node_score[st] = -1.0
            if ORTOOLS_AVAILABLE and goals_states:
                ilp_result = select_path_ilp(
                    start=start_state,
                    goals=goals_states,
                    edges=E,
                    node_score=node_score,
                    budget=steps_remaining,
                    time_limit_sec=2.0,
                    score_scale=1000,
                    discount_factor=0.9,
                    return_states=True,
                )
                if ilp_result is not None:
                    actions, state_sequence = ilp_result
                if self.is_print:
                    if actions:
                        print(f"[ours_graph_ilp] ILP path actions: {actions}")
                    else:
                        print("[ours_graph_ilp] ILP path actions: None")
        else:
            # 3) LLM-based path selection from the constructed graph.
            start_node_id = None
            for node_id, st in id_to_state.items():
                if st == start_state:
                    start_node_id = node_id
                    break
            edge_index = {
                (edge.get("from"), edge.get("to"), str(edge.get("action", "")).upper())
                for edge in edges_data
                if isinstance(edge, dict)
            }
            if start_node_id:
                edge_sequence = select_path_llm(
                    full_graph=full_graph_data,
                    start_node_id=start_node_id,
                    steps_remaining=steps_remaining,
                )
                if len(edge_sequence) > steps_remaining:
                    edge_sequence = []
                if edge_sequence:
                    valid = True
                    current_id = start_node_id
                    actions = []
                    state_sequence = [start_state]
                    for entry in edge_sequence:
                        if not isinstance(entry, dict):
                            valid = False
                            break
                        from_id = entry.get("from")
                        to_id = entry.get("to")
                        action_token = entry.get("action")
                        thought_text = entry.get("thought", "")
                        if (
                            from_id != current_id
                            or not isinstance(to_id, str)
                            or not isinstance(action_token, str)
                        ):
                            valid = False
                            break
                        action_token = action_token.upper()
                        if (from_id, to_id, action_token) not in edge_index:
                            valid = False
                            break
                        next_state = id_to_state.get(to_id)
                        if next_state is None:
                            valid = False
                            break
                        actions.append(action_token)
                        selection_thoughts.append(str(thought_text) if thought_text is not None else "")
                        state_sequence.append(next_state)
                        current_id = to_id
                    if not valid:
                        actions = None
                        state_sequence = None
                        selection_thoughts = []

        if actions and state_sequence:
            cached_thoughts: List[str] = []
            if not self.use_solver and selection_thoughts:
                if len(selection_thoughts) == len(actions):
                    cached_thoughts = selection_thoughts
            if not cached_thoughts:
                for idx, act_token in enumerate(actions):
                    if idx + 1 < len(state_sequence):
                        key = (state_sequence[idx], act_token, state_sequence[idx + 1])
                        cached_thoughts.append(edge_thoughts.get(key, ""))
                    else:
                        cached_thoughts.append("")
            self.cached_actions = actions
            self.cached_state_sequence = state_sequence
            self.cached_thoughts = cached_thoughts
            self.cached_action_index = 0
        else:
            self.cached_actions = []
            self.cached_state_sequence = None
            self.cached_thoughts = None
            self.cached_action_index = 0

        write_graph_log(
            full_graph=full_graph_data,
            merge_log=merge_log_data,
            states=state_list,
            scores=node_score,
            actions=actions,
            state_sequence=state_sequence,
            steps_remaining=steps_remaining,
        )

        def pick_plan_thought() -> str:
            return "Selecting the next action from the planned path."

        def pick_edge_thought(action: str) -> Optional[str]:
            nxt = apply_action(start_state, action, walls)
            if nxt is None:
                return None
            return edge_thoughts.get((start_state, action, nxt))

        # 5) Get next action (first action in the selected path).
        chosen_action = actions[0] if actions else None
        if chosen_action not in {"U", "D", "L", "R"}:
            chosen_action = None

        thought = pick_edge_thought(chosen_action) or pick_plan_thought()
        return render_action_response(chosen_action, thought)

    def observe(self, messages: List[Dict], observation: str):
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        )
        return messages

    def generate_initial_message(self, 
        task_type: str = None, 
        max_examples: int = 0,
        plan_text: str = None
    ) -> str:

        icl_prompt = ""
        examples_text = ""
        if task_type is not None and max_examples >= 0:        
            examples_text, n_examples = self.get_examples_for_task(task_type)
            if min(n_examples, max_examples) > 1:
                icl_prompt = f"Here are {n_examples} examples."
            elif min(n_examples, max_examples) == 1:
                icl_prompt = "Here is an example."

            examples_text = "\n\n".join(examples_text[:min(n_examples, max_examples)]).strip()
        
        template = Template(SOKOBAN_SYSTEM_PROMPT_TEMPLATE)
        self.system_prompt = template.render(
            INSTRUCTION=SOKOBAN_REACT_INSTRUCTION.format(
                sokoban_rule=SOKOBAN_RULE,
                plan_text=plan_text
            ),
            icl_prompt=icl_prompt,
            examples_text=examples_text,
        ).strip()



class OursGraphILPAgentAlfworld(BaseAgent):
    def __init__(
        self,
        model: str,
        fine_tuned_model: Optional[str] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        debug_mode: bool = False,
        instruct_tuned: bool = True,
        enable_lora: bool = False,
        lora=None,
        is_print: bool = False,
        stop: Optional[List[str]] = None,
        method: str = "react",
        num_plans: int = 2,
    ) -> None:
        super().__init__(
            model=model,
            fine_tuned_model=fine_tuned_model,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            debug_mode=debug_mode,
            instruct_tuned=instruct_tuned,
            enable_lora=enable_lora,
            lora=lora,
            is_print=is_print,
        )
        self.client = client
        self.model = model
        self.temperature = temperature
        self.stop = stop or None
        # Predeclare attributes set later to satisfy linters/type checkers
        self.system_prompt = None
        self.user_prompt = None
        self.method = method
        self.plan_text: Optional[str] = None
        self.cached_actions: List[str] = []
        self.cached_state_sequence: Optional[List[GraphState]] = None
        self.cached_thoughts: Optional[List[str]] = None
        self.cached_action_index = 0
        self.graph_log_base: Optional[str] = None
        self.graph_log_index = 0
        self.num_plans = num_plans

    def reset_episode(self) -> None:
        self.plan_text = None
        self.is_plan = False
        self.task_goal = ""
        self.transition_info_text = None
        self.cached_actions = []
        self.cached_state_sequence = None
        self.cached_thoughts = None
        self.cached_action_index = 0
        self.graph_log_index = 0

    def set_graph_log_base(self, base_path: Optional[str]) -> None:
        self.graph_log_base = base_path
        self.graph_log_index = 0

    def get_examples_for_task(self, task_type: str) -> tuple[str, int]:
        """Load ICL examples and concatenate their text with newlines.

        Returns a tuple of (examples_text, num_examples). If none found, returns ("", 0).
        """
        base_dir = os.path.dirname(__file__)
        if self.method == "react":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl.json")
        elif self.method == "react_preplan":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl.json")
        elif self.method == "react_planning":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl_replanning_v2.json")
        elif self.method == "react_formal_planning":
            icl_path = os.path.join(base_dir, "few_shots", "alfworld_icl_replanning_v2.json")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        with open(icl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodes = data.get(task_type, []) or []
        # Each episode is a list of {role, content}; flatten each to lines of content
        episode_texts: List[str] = []
        for ep in episodes:
            contents = [m.get("content", "") for m in (ep or [])]
            # Join messages in an episode with newlines
            episode_texts.append("\n".join(filter(None, contents)))
        assert len(episode_texts) > 0, f"No examples found for task type: {task_type}"
        return episode_texts, len(episode_texts)

    def _is_fixed_temp_model(self, model_name: str) -> bool:
        """Models like o1, o3, and gpt-5 families don't accept temperature.

        Returns True if temperature should be omitted for this model.
        """
        name = (model_name or "").lower()
        return (
            name.startswith("o1")
            or name.startswith("o3")
            or name.startswith("gpt-5")
            or name.startswith("gpt5")
        )

    def _extract_observation(self, text: str) -> Tuple[str, int]:
        steps_match = re.search(r"Steps remaining:\s*(\d+)", text, re.IGNORECASE)
        steps = int(steps_match.group(1)) if steps_match else 0
        if "Observation:" in text:
            obs_part = text.split("Observation:", 1)[1]
            obs_part = obs_part.split("Steps remaining:", 1)[0].strip()
            return obs_part, steps
        return text.strip(), steps

    def generate_plan(
        self,
        observation: str,
        steps_remaining: int,
        transition_info_text: Optional[str] = None,
    ) -> str:
        self.transition_info_text = transition_info_text
        self.task_goal = observation.split("Your task is to: ")[-1].split("\nYou are")[0].strip()
        return None

    def act(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        def normalize_status(status: str) -> str:
            status = (status or "").strip().lower()
            if status in {"heat", "cold", "none"}:
                return status
            return "none"

        def is_coord_state(st: GraphState) -> bool:
            return isinstance(st, GraphState)

        def observation_to_state(obs: Dict[str, Any]) -> StateAlfworld:
            location = str(obs.get("your_location", "") or "").strip()
            observation = str(obs.get("observation", "") or "").strip()
            inventory_raw = obs.get("inventory") or []
            subgoal_progress = str(obs.get("subgoal_progress", "") or "").strip()
            items: List[Tuple[str, str]] = []
            for item in inventory_raw:
                if not isinstance(item, dict):
                    continue
                obj = str(item.get("object", "") or "").strip()
                status = normalize_status(str(item.get("status", "none")))
                if obj:
                    items.append((obj, status))
            items.sort(key=lambda x: (x[0], x[1]))
            return StateAlfworld(tuple(items), location, observation, subgoal_progress)

        def state_to_observation(st: GraphState) -> Optional[Dict[str, Any]]:
            if isinstance(st, StateAlfworld):
                return st.to_observation()
            return None

        def state_to_dict(st: GraphState) -> Dict[str, Any]:
            if is_coord_state(st):
                return {
                    "player": [st.player[0], st.player[1]],
                    "boxes": [[x, y] for x, y in st.boxes],
                }
            obs = state_to_observation(st)
            if obs is None:
                return {"state": str(st)}
            return obs

        def state_sort_key(st: GraphState) -> Tuple[Any, ...]:
            obs = state_to_observation(st)
            if obs is not None:
                inventory = tuple(
                    (item.get("object", ""), item.get("status", "none"))
                    for item in obs.get("inventory", [])
                )
                return (
                    1,
                    str(obs.get("your_location", "")),
                    str(obs.get("observation", "")),
                    inventory,
                )
            if is_coord_state(st):
                return (0, st.player[1], st.player[0], st.boxes)
            return (2, str(st.player), str(st.boxes))

        def write_graph_log(
            *,
            full_graph: Dict[str, Any],
            merge_log: List[Dict[str, Any]],
            states: List[GraphState],
            scores: Dict[GraphState, float],
            actions: Optional[List[str]],
            state_sequence: Optional[List[GraphState]],
            steps_remaining: int,
        ) -> None:
            if not self.graph_log_base:
                return
            self.graph_log_index += 1
            out_path = f"{self.graph_log_base}_graph{self.graph_log_index}.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            score_list = [
                {"state": state_to_dict(st), "score": scores.get(st, 0.0)}
                for st in sorted(states, key=state_sort_key)
            ]
            payload: Dict[str, Any] = {
                "steps_remaining": steps_remaining,
                "graph": full_graph,
                "merge_log": merge_log,
                "node_scores": score_list,
                "actions": actions or [],
            }
            if state_sequence:
                payload["state_sequence"] = [state_to_dict(st) for st in state_sequence]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
        def chat_with_print(
            prompt_messages: List[Dict],
            response_format: Optional[Dict[str, Any]] = None,
            max_tokens: Optional[int] = None,
        ):
            response = self._chat(
                messages=prompt_messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            if self.is_print:
                pretty_print_conversation(
                    prompt_messages
                    + [
                        {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
            return response

        def extract_json(text: str) -> Optional[str]:
            fenced = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, re.DOTALL | re.IGNORECASE)
            if fenced:
                return fenced.group(1)
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            return text[start : end + 1]

        observation_cache: Dict[str, Dict[str, Any]] = {}
        state_compare_cache: Dict[Tuple[str, str], bool] = {}

        observation_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "alfworld_observation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "inventory": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "object": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["heat", "cold", "none"],
                                    },
                                },
                                "required": ["object", "status"],
                                "additionalProperties": False,
                            },
                        },
                        "your_location": {"type": "string"},
                        "observation": {"type": "string"},
                        "subgoal_progress": {"type": "string"},
                    },
                    "required": ["inventory", "your_location", "observation", "subgoal_progress"],
                    "additionalProperties": False,
                },
            },
        }

        state_equal_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "state_equals",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "is_equal": {"type": "boolean"},
                    },
                    "required": ["reason", "is_equal"],
                    "additionalProperties": False,
                },
            },
        }

        def llm_extract_observation(history_text: str) -> Optional[Dict[str, Any]]:
            history_text = (history_text or "").strip()
            if not history_text:
                return None
            cached = observation_cache.get(history_text)
            if cached is not None:
                return cached
            response = chat_with_print(
                [
                    {"role": "system", "content": ALFWORLD_OBSERVATION_EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": ALFWORLD_OBSERVATION_EXTRACT_USER_TEMPLATE.format(history=history_text),
                    },
                ],
                response_format=observation_response_format,
            )
            text = response.choices[0].message.content
            data = None
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                json_blob = extract_json(text)
                if json_blob:
                    data = json.loads(json_blob)
            obs = None
            if isinstance(data, dict):
                if isinstance(data.get("observation"), dict):
                    obs = data["observation"]
                elif "inventory" in data:
                    obs = data
            if obs is not None:
                observation_cache[history_text] = obs
            return obs

        def parse_graph_observation(
            history_text: Optional[str] = None,
        ) -> StateAlfworld:
            source_text = history_text
            obs = llm_extract_observation(source_text)
            return observation_to_state(obs)

        def state_to_compare_payload(st: GraphState) -> Dict[str, Any]:
            return {"inventory": st.inventory, "your_location": st.your_location, "observation": st.observation}

        def states_equal_llm(state_a: GraphState, state_b: GraphState) -> bool:
            payload_a = state_to_compare_payload(state_a)
            payload_b = state_to_compare_payload(state_b)
            response = chat_with_print(
                [
                    {"role": "system", "content": ALFWORLD_STATE_EQUAL_SYSTEM},
                    {
                        "role": "user",
                        "content": ALFWORLD_STATE_EQUAL_USER_TEMPLATE.format(
                            state_a=json.dumps(payload_a, ensure_ascii=True, indent=2),
                            state_b=json.dumps(payload_b, ensure_ascii=True, indent=2),
                        ),
                    },
                ],
                response_format=state_equal_response_format,
            )
            text = response.choices[0].message.content
            data = None
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                json_blob = extract_json(text)
                if json_blob:
                    data = json.loads(json_blob)
            is_equal = None
            if isinstance(data, dict):
                is_equal = data.get("is_equal")
            if isinstance(is_equal, str):
                is_equal = is_equal.strip().lower() == "true"
            result = bool(is_equal) if isinstance(is_equal, bool) else False
            return result

        def score_nodes_llm(
            states: List[GraphState],
            observation_text: str,
            budget: int,
            edges: List[Tuple[GraphState, str, GraphState]] = None,
        ) -> Dict[GraphState, float]:
            if not states:
                return {}
            state_ids = {st: f"node{i}" for i, st in enumerate(states)}
            lines = []
            for st in states:
                sid = state_ids[st]
                lines.append(f"{sid}: {st.to_observation()}")
            
            # Format edges information
            edge_lines = []
            if edges:
                for u, action, v in edges:
                    u_id = state_ids.get(u)
                    v_id = state_ids.get(v)
                    if u_id is not None and v_id is not None:
                        edge_lines.append(f"{u_id} --[{action}]--> {v_id}")
            edges_text = "\n".join(edge_lines) if edge_lines else "No edges available"
            
            system_prompt = ALFWORLD_SCORE_NODES_SYSTEM.format(
                basic_information=self.transition_info_text
            )

            user_prompt = ALFWORLD_SCORE_NODES_USER_TEMPLATE.format(
                task=self.task_goal,
                observation=observation_text,
                edges=edges_text,
                states="\n".join(lines),
            )

            response = chat_with_print(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content
            data = None
            json_blob = extract_json(text)
            if json_blob:
                try:
                    data = json.loads(json_blob)
                except json.JSONDecodeError:
                    data = None
            scores_raw = data.get("scores", data) if isinstance(data, dict) else None
            scores: Dict[GraphState, float] = {}
            if isinstance(scores_raw, dict):
                for key, val in scores_raw.items():
                    st = next((s for s, sid in state_ids.items() if sid == key), None)
                    if st is None:
                        continue
                    try:
                        score = float(val)
                    except (TypeError, ValueError):
                        continue
                    if score <= -0.5:
                        scores[st] = -1.0
                    elif score >= 0.5:
                        scores[st] = 1.0
                    else:
                        scores[st] = 0.0
            return scores

        last_msg = messages[-1]["content"] if messages else ""
        obs_text, steps_remaining = self._extract_observation(last_msg)

        history_text = "\n".join(
            [
                msg.get("content", "")
                for msg in messages[1:]
                if msg.get("content") is not None
            ]
        ).strip()
        current_graph_state = parse_graph_observation(
            history_text=history_text,
        )
        if self.cached_actions and self.cached_state_sequence:
            match_index = None
            start_idx = min(self.cached_action_index, len(self.cached_state_sequence) - 1)
            if states_equal_llm(self.cached_state_sequence[min(start_idx+1, len(self.cached_state_sequence) - 1)], current_graph_state):
                match_index = start_idx + 1
            if match_index is not None and match_index < len(self.cached_actions):
                chosen_action = self.cached_actions[match_index]
                cached_thought = None
                if self.cached_thoughts:
                    cached_thought = self.cached_thoughts[match_index]
                self.cached_action_index = start_idx + 1
                thought_text = cached_thought or "Continuing cached path."
                next_action = f"Action: {chosen_action}" # 
                messages.append({"content": next_action, "role": "assistant"})
                self.logger(messages)
                return [next_action]

            self.cached_actions = []
            self.cached_state_sequence = None
            self.cached_thoughts = None
            self.cached_action_index = 0

        # 1) M plan path sampling. One output should contain M diverse plans.
        plan_system = ALFWORLD_PLAN_INSTRUCTION.format(
            num_plans=self.num_plans,
            basic_information=self.transition_info_text,
        )

        plan_user = ALFWORLD_PLAN_USER_TEMPLATE.format(
            task=self.task_goal,
            observation=history_text, # current_graph_state.to_observation()
            steps_remaining=steps_remaining,
        )

        plan_response = chat_with_print(
            [
                {"role": "system", "content": plan_system},
                {"role": "user", "content": plan_user},
            ],
        )
        plan_text = plan_response.choices[0].message.content

        # 2) Step 1: simulate plans and output per-plan observation sequences.
        graph_step1_system = ALFWORLD_GRAPH_STEP1_SYSTEM.format(
            initial_info=self.transition_info_text, # initial_info
        )
        graph_step1_user = ALFWORLD_GRAPH_STEP1_USER_TEMPLATE.format(
            task=self.task_goal,
            observation=history_text,
            steps_remaining=steps_remaining,
            plans_text=plan_text,
        )

        graph_messages = [
            {"role": "system", "content": graph_step1_system},
            {"role": "user", "content": graph_step1_user},
        ]

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
                                                        "observation": {
                                                            "type": "object",
                                                            "properties": {
                                                                "inventory": {
                                                                    "type": "array",
                                                                    "items": {
                                                                        "type": "object",
                                                                        "properties": {
                                                                            "object": {"type": "string"},
                                                                            "status": {"type": "string"},
                                                                        },
                                                                        "required": ["object", "status"],
                                                                        "additionalProperties": False,
                                                                    },
                                                                },
                                                                "your_location": {"type": "string"},
                                                                "observation": {"type": "string"},
                                                                "subgoal_progress": {"type": "string"}
                                                            },
                                                            "required": [
                                                                "inventory",
                                                                "your_location",
                                                                "observation",
                                                                "subgoal_progress"
                                                            ],
                                                            "additionalProperties": False,
                                                        },
                                                    },
                                                    "required": ["node_id", "thought", "observation"],
                                                    "additionalProperties": False,
                                                },
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "action": {"type": "string"},
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

        graph_step1_response = self._chat(
            messages=graph_messages,
            response_format=graph_step1_response_format,
        )
        graph_step1_text = graph_step1_response.choices[0].message.content
        graph_messages.append(
            {
                "role": graph_step1_response.choices[0].message.role,
                "content": graph_step1_text,
            }
        )
        if self.is_print:
            pretty_print_conversation(graph_messages)

        # 3) Step 2: merge observations and build full graph.
        graph_step2_user = ALFWORLD_GRAPH_STEP2_USER_TEMPLATE.format(
            instructions=ALFWORLD_GRAPH_STEP2_INSTRUCTIONS,
        )

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
                                            "observation": {
                                                "type": "object",
                                                "properties": {
                                                    "inventory": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "object": {"type": "string"},
                                                                "status": {"type": "string"},
                                                            },
                                                            "required": ["object", "status"],
                                                            "additionalProperties": False,
                                                        },
                                                    },
                                                    "your_location": {"type": "string"},
                                                    "observation": {"type": "string"},
                                                    "subgoal_progress": {"type": "string"},
                                                },
                                                "required": [
                                                    "inventory",
                                                    "your_location",
                                                    "observation",
                                                    "subgoal_progress",
                                                ],
                                                "additionalProperties": False,
                                            },
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
                                            "action": {"type": "string"},
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
        graph_messages.append({"role": "user", "content": graph_step2_user})
        graph_step2_response = self._chat(
            messages=graph_messages,
            response_format=graph_step2_response_format,
        )
        graph_step2_text = graph_step2_response.choices[0].message.content
        graph_messages.append(
            {
                "role": graph_step2_response.choices[0].message.role,
                "content": graph_step2_text,
            }
        )
        if self.is_print:
            pretty_print_conversation(graph_messages[-2:])
        graph_step2_data: Optional[Dict[str, Any]] = None
        try:
            graph_step2_data = json.loads(graph_step2_text)
        except json.JSONDecodeError:
            graph_blob = extract_json(graph_step2_text)
            if graph_blob:
                graph_step2_data = json.loads(graph_blob)
        if not isinstance(graph_step2_data, dict):
            raise ValueError("Step 2 response must be a JSON object.")
        merge_log_data = graph_step2_data.get("merge_log")
        full_graph_data = graph_step2_data.get("full_graph")
        if not isinstance(merge_log_data, list):
            raise ValueError("Step 2 response missing merge_log.")
        if not isinstance(full_graph_data, dict):
            raise ValueError("Step 2 response missing full_graph.")
        nodes_data = full_graph_data.get("nodes")
        edges_data = full_graph_data.get("edges")
        if not isinstance(nodes_data, list) or not isinstance(edges_data, list):
            raise ValueError("Step 2 response missing nodes/edges in full_graph.")

        id_to_state: Dict[str, GraphState] = {}
        obs_to_state: Dict[str, GraphState] = {}
        goal_state_ids: List[str] = []
        start_state: GraphState = current_graph_state
        for node in nodes_data:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            observation = node.get("observation")
            is_goal = node.get("is_goal", False)
            if not isinstance(node_id, str):
                continue
            
            st = observation_to_state(observation)
            id_to_state[node_id] = st
            obs_to_state[node_id] = st
            if isinstance(is_goal, bool) and is_goal:
                goal_state_ids.append(node_id)
            
            is_start = node.get("is_start", False)
            if isinstance(is_start, bool) and is_start:
                start_state = st

        V = set(obs_to_state.values())
        E: List[Tuple[GraphState, str, GraphState]] = []
        edge_thoughts: Dict[Tuple[GraphState, str, GraphState], str] = {}
        for edge in edges_data:
            if not isinstance(edge, dict):
                continue
            from_id = edge.get("from")
            to_id = edge.get("to")
            thought = edge.get("thought", "")
            action_token = edge.get("action")
            if (
                not isinstance(from_id, str)
                or not isinstance(to_id, str)
                or not isinstance(action_token, str)
            ):
                continue
            if from_id not in obs_to_state or to_id not in obs_to_state:
                continue
            u = id_to_state.get(from_id)
            v = id_to_state.get(to_id)
            if u is None or v is None:
                continue
            action_token = action_token.lower()
            key = (u, action_token, v)
            E.append(key)
            edge_thoughts[key] = str(thought) if thought is not None else ""
        if start_state not in V:
            V.add(start_state)

        # 3) Graph scoring & constraint prediction (LLM estimates node scores).
        state_list = sorted(V, key=state_sort_key)
        node_score = score_nodes_llm(state_list, current_graph_state.to_observation(), steps_remaining, edges=E)
        if not node_score:
            node_score = {}

        # 4) Run ILP to solve the problem.
        goals_states = {id_to_state[node_id] for node_id in goal_state_ids if node_id in id_to_state}
        out_degree: Dict[GraphState, int] = {st: 0 for st in V}
        for u, _a, _v in E:
            out_degree[u] = out_degree.get(u, 0) + 1
        for st in goals_states:
            node_score[st] = 1.0
        for st in V:
            if st not in goals_states and out_degree.get(st, 0) == 0:
                node_score[st] = -1.0
        actions = None
        state_sequence = None
        if ORTOOLS_AVAILABLE and goals_states:
            ilp_result = select_path_ilp(
                start=start_state,
                goals=goals_states,
                edges=E,
                node_score=node_score,
                budget=steps_remaining,
                time_limit_sec=2.0,
                score_scale=1000,
                discount_factor=0.9,
                return_states=True,
            )
            if ilp_result is not None:
                actions, state_sequence = ilp_result
            if self.is_print:
                if actions:
                    print(f"[ours_graph_ilp] ILP path actions: {actions}")
                else:
                    print("[ours_graph_ilp] ILP path actions: None")
            if actions:
                actions = [action.lower() for action in actions]

        if actions and state_sequence:
            cached_thoughts: List[str] = []
            for idx, act_token in enumerate(actions):
                if idx + 1 < len(state_sequence):
                    key = (state_sequence[idx], act_token, state_sequence[idx + 1])
                    cached_thoughts.append(edge_thoughts.get(key, ""))
                else:
                    cached_thoughts.append("")
            self.cached_actions = actions
            self.cached_state_sequence = state_sequence
            self.cached_thoughts = cached_thoughts
            self.cached_action_index = 0
        else:
            self.cached_actions = []
            self.cached_state_sequence = None
            self.cached_thoughts = None
            self.cached_action_index = 0

        write_graph_log(
            full_graph=full_graph_data,
            merge_log=merge_log_data,
            states=state_list,
            scores=node_score,
            actions=actions,
            state_sequence=state_sequence,
            steps_remaining=steps_remaining,
        )

        # 5) Get next action (first action in the ILP path).
        chosen_action = actions[0] if actions else "look"
        thought = self.cached_thoughts[0] if self.cached_thoughts else "No specific thought."
        next_action = f"Action: {chosen_action}"

        messages.append(
            {
                "content": next_action,
                "role": "assistant",
            }
        )
        self.logger(messages)
        return [next_action]

    def observe(self, messages: List[Dict], observation: str):
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        )
        return messages

    def generate_initial_message(
        self,
        task_type: str = None,
        max_examples: int = 0,
        plan_text: str = None,
        transition_info_text: Optional[str] = None,
    ) -> str:
        icl_prompt = ""
        examples_text = ""
        if task_type is not None and max_examples >= 0:   
            examples_text, n_examples = self.get_examples_for_task(task_type)
            if min(n_examples, max_examples) > 1:
                icl_prompt = f"Here are {n_examples} examples."
            elif min(n_examples, max_examples) == 1:
                icl_prompt = "Here is an example."

            examples_text = "\n\n".join(examples_text[:min(n_examples, max_examples)]).strip()
        
        template = Template(ALFWORLD_SYSTEM_PROMPT_TEMPLATE)
        instruction = ALFWORLD_REACT_INSTRUCTION
        self.system_prompt = template.render(
            INSTRUCTION=instruction,
            icl_prompt=icl_prompt,
            examples_text=examples_text,
        ).strip()
        if transition_info_text:
            self.system_prompt = (
                f"{self.system_prompt}\n\n### Basic Information\n\n {transition_info_text}"
            )

        return self.system_prompt


# ---------------------------------------------------------------------------
# Arithmetic Graph-ILP Agent
# ---------------------------------------------------------------------------

# Lightweight state for arithmetic computation graph nodes.
class ArithState:
    """Hashable computation state for the arithmetic graph."""
    __slots__ = ("observation",)

    def __init__(self, observation: str):
        self.observation = observation

    def __eq__(self, other):
        if not isinstance(other, ArithState):
            return NotImplemented
        return self.observation == other.observation

    def __hash__(self):
        return hash(self.observation)

    def __repr__(self):
        return f"ArithState({self.observation!r})"


class OursGraphILPAgentArithmetic(BaseAgent):
    """Graph-ILP agent for arithmetic tool-use tasks.

    Mirrors OursGraphILPAgent / OursGraphILPAgentAlfworld but:
    - Nodes are computation states (intermediate results + remaining budget).
    - Edges are tool calls (tool_name constrained to an enum + arguments).
    - ILP selects optimal path; cached actions are replayed.
    - Strong-conditioning variant injects solver hint into messages.
    """

    def __init__(
        self,
        model: str,
        fine_tuned_model: Optional[str] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        debug_mode: bool = False,
        instruct_tuned: bool = True,
        enable_lora: bool = False,
        lora=None,
        is_print: bool = False,
        stop: Optional[List[str]] = None,
        num_plans: int = 4,
        use_solver: bool = True,
        use_replanning: bool = True,
        use_strong_conditioning: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            fine_tuned_model=fine_tuned_model,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            debug_mode=debug_mode,
            instruct_tuned=instruct_tuned,
            enable_lora=enable_lora,
            lora=lora,
            is_print=is_print,
        )
        if "gemini" in model:
            self.client = gemini_client
        elif "claude" in model:
            self.client = anthropic_client
        else:
            self.client = client
        self.model = model
        self.temperature = temperature
        self.stop = stop or None
        self.system_prompt = None
        self.num_plans = num_plans
        self.use_solver = use_solver
        self.use_replanning = use_replanning
        self.use_strong_conditioning = use_strong_conditioning

        # Per-episode state
        self.cached_actions: List[Dict[str, Any]] = []  # list of {tool_name, arguments}
        self.cached_state_sequence: Optional[List[ArithState]] = None
        self.cached_thoughts: Optional[List[str]] = None
        self.cached_action_index = 0
        self.graph_log_base: Optional[str] = None
        self.graph_log_index = 0
        self.replanning_log: List[Dict[str, Any]] = []
        # Per-episode online usage stats keyed by canonical tool name.
        self.tool_usage_stats: Dict[str, Dict[str, float]] = {}

        # Set by configure_tools()
        self.tool_names: List[str] = []
        self.tool_descriptions: str = ""
        self.tool_registry: Dict[str, Any] = {}

    # ---- public helpers ----

    def configure_tools(self, tool_registry: Dict[str, Any]) -> None:
        """Must be called before act() to set available tools."""
        self.tool_registry = tool_registry
        self.tool_names = arith_collect_tool_names(tool_registry)
        self.tool_descriptions = arith_tool_description_block(tool_registry)

    def reset_episode(self) -> None:
        self.cached_actions = []
        self.cached_state_sequence = None
        self.cached_thoughts = None
        self.cached_action_index = 0
        self.graph_log_index = 0
        self.replanning_log = []
        self.tool_usage_stats = {}

    def set_graph_log_base(self, base_path: Optional[str]) -> None:
        self.graph_log_base = base_path
        self.graph_log_index = 0
        self.replanning_log = []
        self.tool_usage_stats = {}

    def get_replanning_info(self) -> Dict[str, Any]:
        return {
            "num_replanning": len(self.replanning_log),
            "replanning_states": self.replanning_log,
        }

    def _is_fixed_temp_model(self, model_name: str) -> bool:
        name = (model_name or "").lower()
        return (
            name.startswith("o1")
            or name.startswith("o3")
            or name.startswith("gpt-5")
            or name.startswith("gpt5")
        )

    # ---- observation parsing ----

    def _extract_observation(self, text: str) -> Tuple[str, float, float]:
        """Parse observation text into (obs_text, time_remaining, cost_remaining)."""
        time_match = re.search(r"Time remaining:\s*([\d.]+)", text, re.IGNORECASE)
        cost_match = re.search(r"Cost remaining:\s*\$?([\d.]+)", text, re.IGNORECASE)
        # If a budget dimension is omitted from observation, treat it as unbounded.
        time_rem = float(time_match.group(1)) if time_match else float("inf")
        cost_rem = float(cost_match.group(1)) if cost_match else float("inf")
        return text.strip(), time_rem, cost_rem

    def _extract_question(self, messages: List[Dict]) -> str:
        """Extract the question from the conversation history."""
        for msg in messages:
            content = msg.get("content", "")
            match = re.search(r"Question:\s*(.+?)(?:\n|$)", content)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_computation_history(self, messages: List[Dict]) -> str:
        """Summarize tool results seen so far."""
        results = []
        for msg in messages:
            content = msg.get("content", "")
            for m in re.finditer(r"Tool result:\s*(.+?)(?:\n|$)", content):
                results.append(m.group(1).strip())
        return "; ".join(results) if results else "none"

    # ---- plan / initial message (ReAct interface) ----

    def generate_plan(self, observation: str, steps_remaining: int, **kwargs) -> str:
        return None

    def generate_initial_message(
        self,
        task_type: str = None,
        max_examples: int = 0,
        plan_text: str = None,
        transition_info_text: Optional[str] = None,
        tool_registry: Optional[Dict[str, Any]] = None,
    ) -> str:
        if tool_registry:
            self.configure_tools(tool_registry)
        plan_section = ""
        if plan_text:
            plan_section = f"The solver suggests the following plan:\n{plan_text}\n"
        self.system_prompt = ARITH_REACT_INSTRUCTION_TEMPLATE.format(
            tool_descriptions=self.tool_descriptions,
            plan_section=plan_section,
        )
        return self.system_prompt

    def observe(self, messages: List[Dict], observation: str):
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        return messages

    # ---- replanning helpers ----

    def _parse_last_tool_observation(self, text: str) -> Dict[str, Any]:
        """Parse the environment observation after a tool call.

        When a budget dimension is not present in observation,
        the corresponding remaining value is None.
        """
        result: Dict[str, Any] = {
            "tool_result": None,
            "tool_error": None,
            "step_cost": 0.0,
            "step_time": 0.0,
            "time_remaining": None,
            "cost_remaining": None,
        }
        r_match = re.search(r"Tool result:\s*(.+?)(?:\n|$)", text)
        if r_match:
            result["tool_result"] = r_match.group(1).strip()
        e_match = re.search(r"Tool error:\s*(.+?)(?:\n|$)", text)
        if e_match:
            result["tool_error"] = e_match.group(1).strip()
        cost_match = re.search(r"cost:\s*\$?([\d.]+)", text, re.IGNORECASE)
        if cost_match:
            result["step_cost"] = float(cost_match.group(1))
        time_used_match = re.search(r"time_used:\s*([\d.]+)", text, re.IGNORECASE)
        if time_used_match:
            result["step_time"] = float(time_used_match.group(1))
        tr_match = re.search(r"Time remaining:\s*([\d.]+)", text, re.IGNORECASE)
        if tr_match:
            result["time_remaining"] = float(tr_match.group(1))
        cr_match = re.search(r"Cost remaining:\s*\$?([\d.]+)", text, re.IGNORECASE)
        if cr_match:
            result["cost_remaining"] = float(cr_match.group(1))
        return result

    @staticmethod
    def _parse_state_budgets(
        observation: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract predicted time_remaining and cost_remaining from a graph node observation.

        Returns None for a dimension that is not present in the observation.
        """
        time_rem: Optional[float] = None
        cost_rem: Optional[float] = None
        t_match = re.search(
            r"(?:time_remaining|time remaining)\s*:\s*([\d.]+)",
            observation,
            re.IGNORECASE,
        )
        if t_match:
            try:
                time_rem = float(t_match.group(1))
            except ValueError:
                pass
        c_match = re.search(
            r"(?:cost_remaining|cost remaining)\s*:\s*\$?([\d.]+)",
            observation,
            re.IGNORECASE,
        )
        if c_match:
            try:
                cost_rem = float(c_match.group(1))
            except ValueError:
                pass
        return time_rem, cost_rem

    @staticmethod
    def _budgets_match(
        actual_time: Optional[float],
        actual_cost: Optional[float],
        predicted_time: Optional[float],
        predicted_cost: Optional[float],
        tolerance: float = 0.3,
    ) -> bool:
        """Check if actual remaining budgets are close enough to predicted.

        Comparison is performed only for budget dimensions observed from the
        environment (actual_* is not None). Missing dimensions are skipped.
        """
        # Time check: only when time budget is observed in environment feedback.
        if actual_time is not None:
            if predicted_time is None:
                return False
            if predicted_time > 0:
                if abs(actual_time - predicted_time) / predicted_time > tolerance:
                    return False
            elif actual_time > 0:
                return False

        # Cost check: only when cost budget is observed in environment feedback.
        if actual_cost is not None:
            if predicted_cost is None:
                return False
            if predicted_cost > 0:
                if abs(actual_cost - predicted_cost) / predicted_cost > tolerance:
                    return False
            elif actual_cost > 0:
                return False

        return True

    def _resolve_tool(self, tool_name: str) -> Tuple[Optional[Any], str]:
        """Resolve tool object and canonical name from registry."""
        tool = self.tool_registry.get(tool_name)
        canonical_name = tool_name
        if tool is not None:
            canonical_name = str(getattr(tool, "name", tool_name))
            return tool, canonical_name

        for key, val in self.tool_registry.items():
            name = str(getattr(val, "name", key))
            if tool_name == key or tool_name == name:
                return val, name
        return None, canonical_name

    def record_tool_execution(
        self,
        tool_name: str,
        step_time: Optional[float],
        step_cost: Optional[float],
    ) -> None:
        """Update per-episode empirical time/cost estimate for a tool."""
        _, canonical_name = self._resolve_tool(tool_name)
        stats = self.tool_usage_stats.setdefault(
            canonical_name,
            {
                "time_sum": 0.0,
                "time_count": 0.0,
                "cost_sum": 0.0,
                "cost_count": 0.0,
            },
        )
        if step_time is not None and math.isfinite(step_time) and step_time >= 0:
            stats["time_sum"] += float(step_time)
            stats["time_count"] += 1.0
        if step_cost is not None and math.isfinite(step_cost) and step_cost >= 0:
            stats["cost_sum"] += float(step_cost)
            stats["cost_count"] += 1.0

    def _estimate_tool_usage(self, tool_name: str) -> Tuple[float, float]:
        """Estimate (time, cost) for a tool call using metadata + episode stats."""
        tool, canonical_name = self._resolve_tool(tool_name)
        if tool is None:
            base_time = 1.0
            base_cost = 1.0
        else:
            base_time = float(
                getattr(
                    tool,
                    "execution_time_mu",
                    getattr(tool, "default_execution_time", 1.0),
                )
            )
            base_cost = float(
                getattr(
                    tool,
                    "cost_mu",
                    getattr(tool, "default_cost", 1.0),
                )
            )

        stats = self.tool_usage_stats.get(canonical_name)
        est_time = base_time
        est_cost = base_cost
        if isinstance(stats, dict):
            time_count = float(stats.get("time_count", 0.0))
            cost_count = float(stats.get("cost_count", 0.0))
            if time_count > 0:
                est_time = float(stats.get("time_sum", 0.0)) / time_count
            if cost_count > 0:
                est_cost = float(stats.get("cost_sum", 0.0)) / cost_count
        return max(0.0, est_time), max(0.0, est_cost)

    # ---- main act() ----

    def act(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        def chat_with_print(
            prompt_messages: List[Dict],
            response_format: Optional[Dict[str, Any]] = None,
            max_tokens_inner: Optional[int] = None,
        ):
            response = self._chat(
                messages=prompt_messages,
                response_format=response_format,
                max_tokens=max_tokens_inner,
            )
            if "claude" in self.model:
                content = response.content[0].text
                role = "assistant"
            else:
                role = response.choices[0].message.role
                content = response.choices[0].message.content
            if self.is_print:
                pretty_print_conversation(
                    prompt_messages + [{"role": role, "content": content}]
                )
            return response

        def extract_json_blob(text: str) -> Optional[str]:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            return text[start: end + 1]

        def format_tool_action(tool_name: str, arguments: Dict[str, Any]) -> str:
            args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
            return f"{tool_name}({args_str})"

        def _budget_prompt_bundle(
            time_rem: float,
            cost_rem: float,
        ) -> Dict[str, str]:
            time_enabled = time_rem != float("inf")
            cost_enabled = cost_rem != float("inf")
            budget_parts: List[str] = []
            remaining_parts: List[str] = []
            if time_enabled:
                budget_parts.append(f"Time budget: {time_rem:.3f}s")
                remaining_parts.append(f"Time remaining: {time_rem:.3f}s")
            if cost_enabled:
                budget_parts.append(f"Cost budget: ${cost_rem:.4f}")
                remaining_parts.append(f"Cost remaining: ${cost_rem:.4f}")
            return {
                "budget_lines": " | ".join(budget_parts),
                "remaining_lines": " | ".join(remaining_parts),
            }

        def _fmt_budget_for_log(value: Optional[float], *, is_cost: bool = False) -> str:
            if value is None:
                return ""
            return f"${value:.4f}" if is_cost else f"{value:.3f}"

        def _fmt_budget_pair_for_log(
            t_value: Optional[float],
            c_value: Optional[float],
        ) -> str:
            parts: List[str] = []
            t_str = _fmt_budget_for_log(t_value, is_cost=False)
            c_str = _fmt_budget_for_log(c_value, is_cost=True)
            if t_str:
                parts.append(f"t={t_str}")
            if c_str:
                parts.append(f"c={c_str}")
            return ", ".join(parts) if parts else "no-budget-fields"

        def _build_strong_conditioning_response_format(
            selected_tool_name: str,
        ) -> Dict[str, Any]:
            """Build a response_format schema that forces the LLM to generate
            a structured Thought+Action for the selected tool."""
            # Look up the tool's input keys from registry to constrain arguments
            arg_properties: Dict[str, Any] = {}
            arg_required: List[str] = []
            for _key, tool in self.tool_registry.items():
                tname = getattr(tool, "name", _key)
                if tname == selected_tool_name:
                    tool_inputs = getattr(tool, "inputs", {})
                    if isinstance(tool_inputs, dict):
                        for inp_name in tool_inputs:
                            arg_properties[inp_name] = {"type": "number"}
                            arg_required.append(inp_name)
                    break
            if not arg_properties:
                arg_properties = {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                }
                arg_required = ["a", "b"]

            schema = {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool_name": {
                        "type": "string",
                        "enum": [selected_tool_name],
                    },
                    "arguments": {
                        "type": "object",
                        "properties": arg_properties,
                        "required": arg_required,
                        "additionalProperties": False,
                    },
                },
                "required": ["thought", "tool_name", "arguments"],
                "additionalProperties": False,
            }
            if "claude" in self.model:
                return schema
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "arithmetic_strong_action",
                    "schema": schema,
                },
            }

        def render_action_response(
            tool_name: Optional[str],
            arguments: Dict[str, Any],
            thought: str,
        ) -> List[str]:
            if tool_name is None:
                action_str = "Answer is \\boxed{Unknown}."
            else:
                action_str = format_tool_action(tool_name, arguments)
            next_action = f"Thought: {thought}\nAction: {action_str}"
            if self.use_strong_conditioning and tool_name is not None:
                hint = f"Hint: {thought}\nSuggested action: {action_str}"
                if messages and messages[-1].get("role") == "user":
                    messages[-1]["content"] += f"\n{hint}"
                else:
                    messages.append({"role": "user", "content": hint})
                resp_format = _build_strong_conditioning_response_format(tool_name)
                response = self._chat(
                    messages=messages,
                    response_format=resp_format,
                    max_tokens=max_tokens,
                )
                if "claude" in self.model:
                    raw_text = response.content[0].text
                    role = "assistant"
                else:
                    raw_text = response.choices[0].message.content
                    role = response.choices[0].message.role
                # Parse structured JSON back into Thought/Action text format
                try:
                    parsed = json.loads(raw_text)
                    t = parsed.get("thought", thought)
                    tn = parsed.get("tool_name", tool_name)
                    args = parsed.get("arguments", arguments)
                    content = f"Thought: {t}\nAction: {format_tool_action(tn, args)}"
                except (json.JSONDecodeError, AttributeError):
                    content = raw_text
                messages.append({"content": content, "role": role})
                self.logger(messages)
                return [content]
            messages.append({"content": next_action, "role": "assistant"})
            self.logger(messages)
            return [next_action]

        def write_graph_log(
            *,
            full_graph: Dict[str, Any],
            merge_log: List[Dict[str, Any]],
            states: List[ArithState],
            scores: Dict[ArithState, float],
            actions: Optional[List[Dict[str, Any]]],
            state_sequence: Optional[List[ArithState]],
            time_remaining: float,
            cost_remaining: float,
        ) -> None:
            if not self.graph_log_base:
                return
            self.graph_log_index += 1
            out_path = f"{self.graph_log_base}_graph{self.graph_log_index}.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            score_list = [
                {"state": st.observation, "score": scores.get(st, 0.0)}
                for st in sorted(states, key=lambda s: s.observation)
            ]
            payload: Dict[str, Any] = {
                "time_remaining": time_remaining,
                "cost_remaining": cost_remaining,
                "graph": full_graph,
                "merge_log": merge_log,
                "node_scores": score_list,
                "actions": [
                    {
                        "tool_name": a["tool_name"],
                        "arguments": a.get("arguments", {}),
                        "estimated_time": a.get("estimated_time", 0.0),
                        "estimated_cost": a.get("estimated_cost", 0.0),
                    }
                    for a in (actions or [])
                ],
            }
            if state_sequence:
                payload["state_sequence"] = [st.observation for st in state_sequence]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)

        # ---- Parse current observation ----
        last_msg = messages[-1]["content"] if messages else ""
        obs_text, time_remaining, cost_remaining = self._extract_observation(last_msg)
        budget_prompt = _budget_prompt_bundle(time_remaining, cost_remaining)
        question = self._extract_question(messages)
        computation_history = self._extract_computation_history(messages)

        if time_remaining <= 0 or cost_remaining <= 0:
            next_action = "Thought: Budget exhausted.\nAction: Answer is \\boxed{Unknown}."
            messages.append({"content": next_action, "role": "assistant"})
            self.logger(messages)
            return [next_action]

        # ---- Replay cached actions if available ----
        if self.cached_actions and self.cached_action_index < len(self.cached_actions):
            should_replan = False

            # Compare actual observation with expected state (like ALFWorld)
            if self.use_replanning and self.cached_action_index > 0:
                actual = self._parse_last_tool_observation(last_msg)

                # Check 1: Tool error (unexpected failure) → replan
                if actual.get("tool_error"):
                    should_replan = True
                    if self.is_print:
                        print("[arith_graph_ilp] Replanning: tool error detected")

                # Check 2: Compare predicted vs actual remaining budgets
                if not should_replan and self.cached_state_sequence:
                    expected_idx = min(
                        self.cached_action_index,
                        len(self.cached_state_sequence) - 1,
                    )
                    expected_state = self.cached_state_sequence[expected_idx]
                    pred_time, pred_cost = self._parse_state_budgets(
                        expected_state.observation
                    )
                    actual_time = actual.get("time_remaining")
                    actual_cost = actual.get("cost_remaining")

                    if not self._budgets_match(
                        actual_time, actual_cost, pred_time, pred_cost
                    ):
                        should_replan = True
                        if self.is_print:
                            print(
                                f"[arith_graph_ilp] Replanning: budget deviation "
                                f"(actual {_fmt_budget_pair_for_log(actual_time, actual_cost)}, "
                                f"predicted {_fmt_budget_pair_for_log(pred_time, pred_cost)})"
                            )

            if not should_replan:
                cached_act = self.cached_actions[self.cached_action_index]
                cached_thought = ""
                if self.cached_thoughts and self.cached_action_index < len(self.cached_thoughts):
                    cached_thought = self.cached_thoughts[self.cached_action_index]
                self.cached_action_index += 1
                return render_action_response(
                    cached_act.get("tool_name"),
                    cached_act.get("arguments", {}),
                    cached_thought or "Continuing cached path.",
                )

            # Mismatch detected → clear cache, fall through to replanning
            self.cached_actions = []
            self.cached_state_sequence = None
            self.cached_thoughts = None
            self.cached_action_index = 0

        # ---- (Re)planning: build graph via LLM ----
        self.cached_actions = []
        self.cached_state_sequence = None
        self.cached_thoughts = None
        self.cached_action_index = 0

        self.replanning_log.append({
            "time_remaining": time_remaining,
            "cost_remaining": cost_remaining,
            "computation_history": computation_history,
        })

        # 1) Generate M diverse plans
        plan_system = ARITH_PLAN_INSTRUCTION_TEMPLATE.format(
            tool_descriptions=self.tool_descriptions,
            num_plans=self.num_plans,
        )
        plan_user = ARITH_PLAN_USER_TEMPLATE.format(
            question=question,
            budget_lines=budget_prompt["budget_lines"],
            remaining_lines=budget_prompt["remaining_lines"],
            computation_history=computation_history,
        )
        plan_response = chat_with_print([
            {"role": "system", "content": plan_system},
            {"role": "user", "content": plan_user},
        ])
        if "claude" in self.model:
            plan_text = plan_response.content[0].text
        else:
            plan_text = plan_response.choices[0].message.content

        # 2) Step 1: simulate plans → per-plan step sequences
        graph_step1_system = ARITH_GRAPH_STEP1_SYSTEM_TEMPLATE.format(
            tool_descriptions=self.tool_descriptions,
        )
        graph_step1_user = ARITH_GRAPH_STEP1_USER_TEMPLATE.format(
            question=question,
            budget_lines=budget_prompt["budget_lines"],
            remaining_lines=budget_prompt["remaining_lines"],
            computation_history=computation_history,
            plans_text=plan_text,
        )
        graph_messages = [
            {"role": "system", "content": graph_step1_system},
            {"role": "user", "content": graph_step1_user},
        ]
        if "claude" in self.model:
            step1_format = arith_build_graph_step1_response_format_claude(self.tool_names)
        else:
            step1_format = arith_build_graph_step1_response_format(self.tool_names)
        graph_step1_resp = self._chat(
            messages=graph_messages,
            response_format=step1_format,
        )
        if "claude" in self.model:
            graph_step1_text = graph_step1_resp.content[0].text
            graph_step1_role = "assistant"
        else:
            graph_step1_text = graph_step1_resp.choices[0].message.content
            graph_step1_role = graph_step1_resp.choices[0].message.role
        graph_messages.append({"role": graph_step1_role, "content": graph_step1_text})
        if self.is_print:
            pretty_print_conversation(graph_messages)

        # 3) Step 2: merge nodes → full graph
        graph_step2_user = ARITH_GRAPH_STEP2_USER_TEMPLATE.format(
            instructions=ARITH_GRAPH_STEP2_INSTRUCTIONS,
        )
        graph_messages.append({"role": "user", "content": graph_step2_user})
        if "claude" in self.model:
            step2_format = arith_build_graph_step2_response_format_claude(self.tool_names)
        else:
            step2_format = arith_build_graph_step2_response_format(self.tool_names)
        graph_step2_resp = self._chat(
            messages=graph_messages,
            response_format=step2_format,
        )
        if "claude" in self.model:
            graph_step2_text = graph_step2_resp.content[0].text
        else:
            graph_step2_text = graph_step2_resp.choices[0].message.content
        graph_messages.append({
            "role": "assistant",
            "content": graph_step2_text,
        })
        if self.is_print:
            pretty_print_conversation(graph_messages[-2:])

        # Parse graph JSON
        graph_step2_data: Optional[Dict[str, Any]] = None
        try:
            graph_step2_data = json.loads(graph_step2_text)
        except json.JSONDecodeError:
            blob = extract_json_blob(graph_step2_text)
            if blob:
                try:
                    graph_step2_data = json.loads(blob)
                except json.JSONDecodeError:
                    pass
        if not isinstance(graph_step2_data, dict):
            # Fallback: no graph, take random action
            fallback = f"Thought: Graph construction failed.\nAction: Answer is \\boxed{{Unknown}}."
            messages.append({"content": fallback, "role": "assistant"})
            self.logger(messages)
            return [fallback]

        merge_log_data = graph_step2_data.get("merge_log", [])
        full_graph_data = graph_step2_data.get("full_graph", {})
        nodes_data = full_graph_data.get("nodes", [])
        edges_data = full_graph_data.get("edges", [])

        # Build internal structures
        id_to_state: Dict[str, ArithState] = {}
        goal_state_ids: List[str] = []
        start_state: Optional[ArithState] = None
        for node in nodes_data:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            obs = node.get("observation", "")
            if not isinstance(node_id, str):
                continue
            st = ArithState(str(obs))
            id_to_state[node_id] = st
            if node.get("is_goal", False):
                goal_state_ids.append(node_id)
            if node.get("is_start", False) and start_state is None:
                start_state = st
        if start_state is None and id_to_state:
            start_state = next(iter(id_to_state.values()))

        V = set(id_to_state.values())
        E: List[Tuple[ArithState, str, ArithState]] = []
        edge_tool_calls: Dict[Tuple[ArithState, str, ArithState], Dict[str, Any]] = {}
        edge_time_est: Dict[Tuple[ArithState, str, ArithState], float] = {}
        edge_cost_est: Dict[Tuple[ArithState, str, ArithState], float] = {}
        edge_thoughts: Dict[Tuple[ArithState, str, ArithState], str] = {}
        for edge in edges_data:
            if not isinstance(edge, dict):
                continue
            from_id = edge.get("from")
            to_id = edge.get("to")
            tool_name = edge.get("tool_name", "")
            arguments = edge.get("arguments", {})
            thought = edge.get("thought", "")
            if not isinstance(from_id, str) or not isinstance(to_id, str):
                continue
            u = id_to_state.get(from_id)
            v = id_to_state.get(to_id)
            if u is None or v is None:
                continue
            # Use tool_name as the "action label" for ILP edges
            action_label = tool_name
            key = (u, action_label, v)
            E.append(key)
            expected_result = edge.get("expected_result", "")
            est_time, est_cost = self._estimate_tool_usage(tool_name)
            src_time, src_cost = self._parse_state_budgets(u.observation)
            dst_time, dst_cost = self._parse_state_budgets(v.observation)
            if src_time is not None and dst_time is not None:
                delta_t = src_time - dst_time
                if delta_t >= 0:
                    est_time = delta_t
            if src_cost is not None and dst_cost is not None:
                delta_c = src_cost - dst_cost
                if delta_c >= 0:
                    est_cost = delta_c
            edge_tool_calls[key] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "expected_result": str(expected_result) if expected_result else "",
                "estimated_time": float(est_time),
                "estimated_cost": float(est_cost),
            }
            edge_time_est[key] = float(est_time)
            edge_cost_est[key] = float(est_cost)
            edge_thoughts[key] = str(thought) if thought else ""
        if start_state and start_state not in V:
            V.add(start_state)

        # 4) Score nodes
        state_list = sorted(V, key=lambda s: s.observation)
        node_score: Dict[ArithState, float] = {}
        if self.use_solver and V and E:
            state_ids_map = {st: f"n{i}" for i, st in enumerate(state_list)}
            score_lines = []
            for st in state_list:
                sid = state_ids_map[st]
                score_lines.append(f"{sid}: {st.observation}")
            edge_lines = []
            for u, action_label, v in E:
                uid = state_ids_map.get(u, "?")
                vid = state_ids_map.get(v, "?")
                tc = edge_tool_calls.get((u, action_label, v), {})
                edge_lines.append(
                    f"{uid} --[{tc.get('tool_name', action_label)}]--> {vid}"
                )
            score_response = chat_with_print([
                {"role": "system", "content": ARITH_SCORE_NODES_SYSTEM},
                {"role": "user", "content": ARITH_SCORE_NODES_USER_TEMPLATE.format(
                    question=question,
                    remaining_lines=budget_prompt["remaining_lines"],
                    edges="\n".join(edge_lines) if edge_lines else "No edges",
                    states="\n".join(score_lines),
                )},
            ])
            if "claude" in self.model:
                score_text = score_response.content[0].text
            else:
                score_text = score_response.choices[0].message.content
            score_blob = extract_json_blob(score_text)
            if score_blob:
                try:
                    score_data = json.loads(score_blob)
                except json.JSONDecodeError:
                    score_data = None
            else:
                score_data = None
            if isinstance(score_data, dict):
                scores_raw = score_data.get("scores", score_data)
                if isinstance(scores_raw, dict):
                    for key, val in scores_raw.items():
                        st = next(
                            (s for s, sid in state_ids_map.items() if sid == key),
                            None,
                        )
                        if st is None:
                            continue
                        try:
                            score = float(val)
                        except (TypeError, ValueError):
                            continue
                        if score <= -0.5:
                            node_score[st] = -1.0
                        elif score >= 0.5:
                            node_score[st] = 1.0
                        else:
                            node_score[st] = 0.0

        # Force goal/dead-end scores
        goals_states = {id_to_state[nid] for nid in goal_state_ids if nid in id_to_state}
        out_degree: Dict[ArithState, int] = {st: 0 for st in V}
        for u, _a, _v in E:
            out_degree[u] = out_degree.get(u, 0) + 1
        for st in goals_states:
            node_score[st] = 1.0
        for st in V:
            if st not in goals_states and out_degree.get(st, 0) == 0:
                node_score[st] = -1.0

        # 5) ILP path selection
        actions_path: Optional[List[str]] = None
        state_sequence: Optional[List[ArithState]] = None
        step_budget_limit = 20
        time_budget_limit = (
            None if time_remaining == float("inf") else float(max(0.0, time_remaining))
        )
        # If explicit cost remaining is absent, use the existing step budget as
        # a fallback resource cap so unit-cost edges still obey a global budget.
        cost_budget_limit = (
            float(step_budget_limit)
            if cost_remaining == float("inf")
            else float(max(0.0, cost_remaining))
        )
        if ORTOOLS_AVAILABLE and goals_states and start_state and E:
            ilp_result = select_path_ilp(
                start=start_state,
                goals=goals_states,
                edges=E,
                node_score=node_score,
                budget=step_budget_limit,  # max steps
                time_limit_sec=2.0,
                score_scale=1000,
                discount_factor=0.9,
                return_states=True,
                edge_time=edge_time_est,
                edge_cost=edge_cost_est,
                time_budget=time_budget_limit,
                cost_budget=cost_budget_limit,
            )
            if ilp_result is not None:
                actions_path, state_sequence = ilp_result
            if self.is_print:
                if actions_path:
                    print(f"[arith_graph_ilp] ILP path actions: {actions_path}")
                else:
                    print("[arith_graph_ilp] ILP path: None")

        # Cache actions
        if actions_path and state_sequence:
            cached_tool_calls: List[Dict[str, Any]] = []
            cached_thoughts_list: List[str] = []
            for idx, act_label in enumerate(actions_path):
                if idx + 1 < len(state_sequence):
                    key = (state_sequence[idx], act_label, state_sequence[idx + 1])
                    tc = dict(edge_tool_calls.get(key, {"tool_name": act_label, "arguments": {}}))
                else:
                    tc = {"tool_name": act_label, "arguments": {}}
                cached_tool_calls.append(tc)
                if idx + 1 < len(state_sequence):
                    key = (state_sequence[idx], act_label, state_sequence[idx + 1])
                    cached_thoughts_list.append(edge_thoughts.get(key, ""))
                else:
                    cached_thoughts_list.append("")
            self.cached_actions = cached_tool_calls
            self.cached_state_sequence = state_sequence
            self.cached_thoughts = cached_thoughts_list
            self.cached_action_index = 0
        else:
            self.cached_actions = []
            self.cached_state_sequence = None
            self.cached_thoughts = None
            self.cached_action_index = 0

        write_graph_log(
            full_graph=full_graph_data,
            merge_log=merge_log_data if isinstance(merge_log_data, list) else [],
            states=state_list,
            scores=node_score,
            actions=self.cached_actions if self.cached_actions else None,
            state_sequence=state_sequence,
            time_remaining=time_remaining,
            cost_remaining=cost_remaining,
        )

        # 6) Return first action
        if self.cached_actions:
            first_tc = self.cached_actions[0]
            first_thought = self.cached_thoughts[0] if self.cached_thoughts else ""
            self.cached_action_index = 1
            return render_action_response(
                first_tc.get("tool_name"),
                first_tc.get("arguments", {}),
                first_thought or "Executing first step of the ILP-selected path.",
            )

        # No path found
        fallback = "Thought: No solution path found.\nAction: Answer is \\boxed{Unknown}."
        messages.append({"content": fallback, "role": "assistant"})
        self.logger(messages)
        return [fallback]

import re
from typing import Any, Dict, List, Optional

from jinja2 import Template

from real_agents.base import BaseAgent
from real_agents.prompts.react_prompts import (
    REACT_INSTRUCTION,
    REACT_INSTRUCTION_ALFWORLD,
    RULE,
    SYSTEM_PROMPT_TEMPLATE,
)
from real_agents.prompts.arithmetic_prompts import build_arithmetic_system_prompt
from utils.llm import anthropic_client, client, gemini_client


class ReACTAgent(BaseAgent):
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
        task: str = "sokoban",
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
        self.user_prompt = None
        self.method = method
        self.task = task

    def _is_fixed_temp_model(self, model_name: str) -> bool:
        """Models like o1, o3, and gpt-5 families don't accept temperature."""
        name = (model_name or "").lower()
        return (
            name.startswith("o1")
            or name.startswith("o3")
            or name.startswith("gpt-5")
            or name.startswith("gpt5")
        )

    def _normalize_action(self, action: str) -> str:
        return " ".join(action.strip().strip("\"'").split())

    def _extract_action(self, text: str) -> Optional[str]:
        if not text:
            return None
        matches = list(re.finditer(r"Action\s*:\s*", text, re.IGNORECASE))
        if not matches:
            return None
        last = matches[-1]
        remainder = text[last.end():]
        for line in remainder.splitlines():
            line = line.strip()
            if line:
                return self._normalize_action(line)
        return None

    def _pick_best_response_index(self, candidates: List[str]) -> int:
        if not candidates:
            return 0

        action_counts: Dict[str, int] = {}
        first_idx: Dict[str, int] = {}
        for idx, content in enumerate(candidates):
            action = self._extract_action(content)
            if action is None:
                continue
            if action not in action_counts:
                action_counts[action] = 0
                first_idx[action] = idx
            action_counts[action] += 1

        if not action_counts:
            return 0

        best_action = max(
            action_counts.keys(),
            key=lambda act: (action_counts[act], -first_idx[act]),
        )
        return first_idx[best_action]

    def generate_plan(
        self,
        observation: str,
        steps_remaining: int,
        transition_info_text: Optional[str] = None,
    ) -> str:
        _ = observation, steps_remaining, transition_info_text
        return None

    def generate_initial_message(
        self,
        task_type: str = None,
        max_examples: int = 0,
        plan_text: str = None,
        transition_info_text: Optional[str] = None,
        tool_registry: Optional[Dict[str, Any]] = None,
    ) -> str:
        # few_shots 기반 ICL은 제거했고, task_type/max_examples는 하위호환을 위해 유지.
        _ = task_type, max_examples, plan_text

        if self.task == "arithmetic":
            self.system_prompt = build_arithmetic_system_prompt(tool_registry or {})
        else:
            template = Template(SYSTEM_PROMPT_TEMPLATE)
            if self.task == "alfworld":
                instruction = REACT_INSTRUCTION_ALFWORLD
            else:
                instruction = REACT_INSTRUCTION.format(sokoban_rule=RULE)

            self.system_prompt = template.render(INSTRUCTION=instruction).strip()
        if transition_info_text:
            self.system_prompt = (
                f"{self.system_prompt}\n\n### Basic Information\n\n {transition_info_text}"
            )
        return self.system_prompt

    def act(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = 10000,
        n: int = 1,
    ):
        _ = max_tokens
        response = self._chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            n=n,
        )

        if "claude" in self.model:
            if isinstance(response, list):
                candidate_contents = [r.content[0].text for r in response]
            else:
                candidate_contents = [response.content[0].text]
            candidate_roles = ["assistant"] * len(candidate_contents)
        else:
            candidate_contents = [choice.message.content for choice in response.choices]
            candidate_roles = [choice.message.role for choice in response.choices]

        best_idx = self._pick_best_response_index(candidate_contents)
        content = candidate_contents[best_idx] if candidate_contents else ""
        role = candidate_roles[best_idx] if candidate_roles else "assistant"

        messages.append(
            {
                "content": content,
                "role": role,
            }
        )
        self.logger(messages)

        if candidate_contents:
            if best_idx == 0:
                return candidate_contents
            return [candidate_contents[best_idx]] + [
                candidate_contents[i]
                for i in range(len(candidate_contents))
                if i != best_idx
            ]
        return [content]

    def observe(self, messages: List[Dict], observation: str):
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        )
        return messages

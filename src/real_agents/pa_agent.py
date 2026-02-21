from typing import Dict, List, Optional

from jinja2 import Template

from real_agents.prompts.pa_prompts import (
    PA_PLAN_INSTRUCTION,
    PA_PLAN_INSTRUCTION_ALFWORLD,
    PA_PLAN_INSTRUCTION_ARITHMETIC,
    PA_REACT_INSTRUCTION,
    PA_REACT_INSTRUCTION_ALFWORLD,
    PA_REACT_INSTRUCTION_ARITHMETIC,
)
from real_agents.prompts.react_prompts import RULE, SYSTEM_PROMPT_TEMPLATE
from real_agents.react_agent import ReACTAgent


class PAAgent(ReACTAgent):
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
            stop=stop,
            method=method,
            task=task,
        )
        self.plan_text: Optional[str] = None

    def reset_episode(self) -> None:
        self.plan_text = None
        self.is_plan = False

    def generate_plan(
        self,
        observation: str,
        steps_remaining: int,
        transition_info_text: Optional[str] = None,
    ) -> str:
        if self.task == "arithmetic":
            tool_descriptions = transition_info_text or "- (no tools available)"
            system_prompt = PA_PLAN_INSTRUCTION_ARITHMETIC.format(
                tool_descriptions=tool_descriptions
            )
        elif self.task == "alfworld":
            system_prompt = PA_PLAN_INSTRUCTION_ALFWORLD
            if transition_info_text:
                system_prompt = (
                    f"{system_prompt}\n\n### Basic Information\n\n {transition_info_text}"
                )
        else:
            system_prompt = PA_PLAN_INSTRUCTION.format(sokoban_rule=RULE)

        user_prompt = f"Observation:\n{observation}\nSteps remaining: {steps_remaining}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._chat(messages=messages)
        if "claude" in self.model:
            content = response.content[0].text
        else:
            content = response.choices[0].message.content

        self.plan_text = content
        self.is_plan = True
        return content

    def generate_initial_message(
        self,
        task_type: str = None,
        max_examples: int = 0,
        plan_text: str = None,
        transition_info_text: Optional[str] = None,
    ) -> str:
        # few_shots 기반 ICL은 제거했고, task_type/max_examples는 하위호환을 위해 유지.
        _ = task_type, max_examples

        template = Template(SYSTEM_PROMPT_TEMPLATE)
        if self.task == "arithmetic":
            tool_descriptions = transition_info_text or "- (no tools available)"
            instruction = PA_REACT_INSTRUCTION_ARITHMETIC.format(
                tool_descriptions=tool_descriptions,
                plan_text=plan_text or "",
            )
        elif self.task == "alfworld":
            instruction = PA_REACT_INSTRUCTION_ALFWORLD.format(
                plan_text=plan_text or "",
            )
        else:
            instruction = PA_REACT_INSTRUCTION.format(
                sokoban_rule=RULE,
                plan_text=plan_text or "",
            )

        self.system_prompt = template.render(INSTRUCTION=instruction).strip()
        if transition_info_text and self.task != "arithmetic":
            self.system_prompt = (
                f"{self.system_prompt}\n\n### Basic Information\n\n {transition_info_text}"
            )
        return self.system_prompt

from transformers import AutoModelForCausalLM
from utils.llm import (
    Logger,
    anthropic_client,
    client,
    gemini_client,
    pretty_print_conversation,
)

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
# Base Agent

class BaseAgent:
    def __init__(
        self,
        model: str,
        fine_tuned_model: str,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        debug_mode: bool = False,
        instruct_tuned: bool = True,
        enable_lora: bool = False,
        lora: AutoModelForCausalLM | None = None,
        is_print: bool = False,
    ):
        # Define and setup llm
        self.orig_model = model
        self.fine_tuned_model = fine_tuned_model
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.instruct_tuned = instruct_tuned
        self.enable_lora = enable_lora
        self.lora = lora

        self.debug_mode = debug_mode
        self.is_print = is_print
        self.model = fine_tuned_model or model
        self.stop = None
        if "gemini" in self.model:
            self.client = gemini_client
        elif "claude" in self.model:
            self.client = anthropic_client
        else:
            self.client = client

        self.log_root_path = "logs/"
        agent_name = self.__class__.__name__
        self.logger = Logger(
			agent_name=agent_name,
			log_root_path=self.log_root_path,
		)

    def update_parameters(
        self,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        if temperature is not None:
            self.temperature = temperature
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty

    def update_model(
        self, new_model: str, new_fine_tuned_model: str | None = None
    ) -> None:
        self.orig_model = new_model
        self.fine_tuned_model = new_fine_tuned_model
        self.model = new_fine_tuned_model or new_model
        if "gemini" in self.model:
            self.client = gemini_client
        elif "claude" in self.model:
            self.client = anthropic_client
        else:
            self.client = client

    def call_llm(
        self, 
        system_prompt, 
        user_prompt,
        n=1,
        response_format=None
    ):
        messages: List[Dict[str, Any]] = []

        if len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        if self.enable_lora:
            raise NotImplementedError(
                "LoRA path is not supported in BaseAgent.call_llm after llm refactor."
            )
        response = self._chat(
            messages=messages,
            n=n,
            response_format=response_format,
        )

        if "claude" in self.model:
            if isinstance(response, list):
                candidate_contents = [r.content[0].text for r in response]
                candidate_roles = ["assistant"] * len(candidate_contents)
            else:
                candidate_contents = [response.content[0].text]
                candidate_roles = ["assistant"]
        else:
            candidate_contents = [choice.message.content for choice in response.choices]
            candidate_roles = [choice.message.role for choice in response.choices]

        messages.append(
            {
                "content": candidate_contents[0] if candidate_contents else "",
                "role": candidate_roles[0] if candidate_roles else "assistant",
            }
        )
        if self.is_print:
            pretty_print_conversation(messages)
            
        self.logger(messages)
        return candidate_contents

    def call_llm_messages(
        self,
        messages: list[dict],
        n: int = 1,
        response_format=None,
        max_tokens: int | None = None,
    **kwargs,
    ):
        """Call LLM with a pre-built multi-turn conversation.

        Args:
            messages: List of dicts with keys {"role", "content"}, e.g.,
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
            n: number of completions to request.
            response_format: optional response format passthrough.
            max_tokens: optional maximum tokens for the completion.

        Returns:
            List[str] of assistant predictions (one per choice), consistent with call_llm.
        """
        # Ensure basic structure
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("messages must be a list of dicts with 'role' and 'content'")

        if self.enable_lora:
            raise NotImplementedError(
                "LoRA path is not supported in BaseAgent.call_llm_messages after llm refactor."
            )
        response = self._chat(
            messages=messages,
            n=n,
            response_format=response_format,
            max_tokens=max_tokens,
        )

        if "claude" in self.model:
            if isinstance(response, list):
                candidate_contents = [r.content[0].text for r in response]
                candidate_roles = ["assistant"] * len(candidate_contents)
            else:
                candidate_contents = [response.content[0].text]
                candidate_roles = ["assistant"]
        else:
            candidate_contents = [choice.message.content for choice in response.choices]
            candidate_roles = [choice.message.role for choice in response.choices]

        # Append assistant reply for logging/pretty print
        messages.append(
            {
                "content": candidate_contents[0] if candidate_contents else "",
                "role": candidate_roles[0] if candidate_roles else "assistant",
            }
        )

        if self.is_print:
            pretty_print_conversation(messages)

        self.logger(messages)
        return candidate_contents

    def _chat(
        self,
        *,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """Unified chat call that adjusts kwargs by model capability.

        - Omits temperature for o1/o3/gpt-5 families which don't support it.
        - Includes tools and tool_choice when provided.
        - Passes stop tokens when configured.
        - Supports n>=1 for multi-sample requests when the backend allows it.
        """
        if n is None or n < 1:
            n = 1
        returned_message = deepcopy(messages)
        if "claude" in self.model:
            system_message = messages[0]["content"] if messages[0]["role"] == "system" else None
            if system_message:
                returned_message = messages[1:]
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "system": system_message,
                "messages": returned_message,
            }
        else:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": returned_message,
            }

        if not self._is_fixed_temp_model(self.model) and self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if self.stop:
            kwargs["stop"] = self.stop

        if tools:
            kwargs["tools"] = tools

        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_format is not None:
            if "claude" in self.model:
                anthropic_output_format = {
                    "type": "json_schema",
                    "schema": response_format,
                }

                kwargs["output_format"] = anthropic_output_format
                kwargs["betas"] = ["structured-outputs-2025-11-13"]
            else:
                kwargs["response_format"] = response_format

        if "gemini" in self.model:
           kwargs["reasoning_effort"] = "low"

        if "claude" in self.model:
            kwargs["max_tokens"] = 20000
            if n > 1:
                return [self.client.beta.messages.create(**kwargs) for _ in range(n)]
            return self.client.beta.messages.create(**kwargs)
        else:
            kwargs["n"] = n
            if response_format is not None:
                try:
                    return self.client.beta.chat.completions.parse(**kwargs)
                except Exception:
                    return self.client.chat.completions.create(**kwargs)
            return self.client.chat.completions.create(**kwargs)

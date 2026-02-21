import os
import re
from typing import Any, Iterable, List

import numpy as np
import requests
import tiktoken
from anthropic.types import MessageParam
# from attrdict import AttrDict
import addict
from openai import OpenAI, Stream
from openai.types import Completion
from transformers import AutoTokenizer

from llms.anthropic_utils import (
    chat_completion_request as anthropic_chat_completion_request,
)
from llms.openai_utils import (
    MoneyManager,
    chat_completion_request,
    completion_request,
)
from llms.gemini_utils import (
    chat_completion_request as chat_completion_request_gemini,
    completion_request as completion_request_gemini,
)
from llms.typing import CompletionFunc, Message


class FunctionChatError(Exception):
    """Raised when a function-enabled chat follow-up request fails."""


def load_model(
    model: str,
    fine_tuned_model: str | None = None,
    temperature: float = 1.0,
    instruct_tuned: bool = True,
    repetition_penalty: float = 0,
) -> dict[str, Any]:
    if "gpt-3.5" in model:
        default_model = model
        if "16k" in model:
            output_budget = 4096
        else:
            output_budget = 1024
        model_type = "chatgpt"
    elif "gpt-4" in model or "o1" in model or "o3" in model or "o4" in model or "gpt-5" in model:
        default_model = model
        output_budget = 64000
        model_type = "chatgpt"
    elif "claude" in model:
        default_model = model
        output_budget = 64000
        model_type = "claude"
    elif "gemini" in model:
        default_model = model
        output_budget = 64000
        model_type = "gemini"
        print("[HERE] gemini")
    else:
        default_model = model
        model_type = "local"
        output_budget = 1024

    if fine_tuned_model is None:
        model = default_model
    else:
        model = fine_tuned_model

    # Main model
    ctx_manager = MoneyManager(model=model)

    if model_type == "chatgpt":
        llm = ChatGPTBase(
            model=model,
            ctx_manager=ctx_manager,
            desired_output_length=output_budget,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        enc = tiktoken.get_encoding("cl100k_base")
    elif model_type == "claude":
        llm = ClaudeBase(
            model=model,
            ctx_manager=ctx_manager,
            desired_output_length=output_budget,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        enc = tiktoken.get_encoding("cl100k_base")
    elif model_type == "gemini":
        llm = GeminiBase(
            model=model,
            ctx_manager=ctx_manager,
            desired_output_length=output_budget,
            temperature=temperature,
        )
        enc = tiktoken.get_encoding("cl100k_base")
    elif model_type == "local":
        llm = LocalBase(
            model=model,
            ctx_manager=ctx_manager,
            desired_output_length=output_budget,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            instruct_tuned=instruct_tuned,
        )
        enc = tiktoken.get_encoding("cl100k_base")  # llm.engine.get_tokenizer()
    else:
        raise NotImplementedError

    return {
        "model_name": model,
        "llm": llm,
        "tokenizer": enc,
        "ctx_manager": ctx_manager,
    }


# ChatGPT having tools
class ChatGPTBase:
    def __init__(
        self,
        model: str,
        tool: Iterable[CompletionFunc] | None = None,
        ctx_manager: MoneyManager | None = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.tool = tool
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.enc = tiktoken.get_encoding("cl100k_base")
        if "gpt-3.5" in self.model:
            if "16k" in self.model or "1106" in self.model:
                self.max_budget = 16384
            else:
                self.max_budget = 4096
        elif "gpt-4-1106-preview" in self.model:
            self.max_budget = 128000
        elif "gpt-4" in self.model:
            self.max_budget = 128000
        elif "o1" in self.model:
            self.max_budget = 128000
        elif "o3" in self.model:
            self.max_budget = 128000
        elif "o4" in self.model:
            self.max_budget = 128000
        elif "gpt-5" in self.model:
            self.max_budget = 128000
        else:
            raise NotImplementedError()
        self.desired_output_length = desired_output_length
        self.temperature = temperature
        # keep as float; Local models may use it, OpenAI Chat models ignore it
        self.repetition_penalty = float(repetition_penalty)

    def _supports_sampling_params(self) -> bool:
        name = self.model
        return not ("gpt-5" in name or any(x in name for x in ("o1", "o3", "o4")))

    def cutoff(self, message: str, budget: int) -> str:
        tokens = self.enc.encode(message)
        if len(tokens) > budget:
            message = self.enc.decode(tokens[:budget])
        return message

    def manage_length(self, messages: List[Message]) -> None:
        last_message = messages[-1]["content"]
        if len(messages) > 1:
            previous_tokens_length = 0
            for msg in messages[:-1]:
                if "content" in msg.keys() and msg["content"] is not None:
                    previous_tokens_length += len(
                        self.enc.encode(msg["content"])
                    )
                elif (
                    "function_call" in msg.keys()
                    and msg["function_call"] is not None
                ):
                    previous_tokens_length += (
                        len(self.enc.encode(msg["function_call"]["arguments"]))
                        + 30
                    )
        else:
            previous_tokens_length = 0
        budget = (
            self.max_budget
            - self.desired_output_length
            - previous_tokens_length
        )
        messages[-1]["content"] = self.cutoff(last_message, budget)

    def chat(
        self,
        messages: List[Message],
        disable_function: bool = False,
        **kwargs,
    ):
        self.manage_length(messages)
        req_kwargs = dict(kwargs)
        if self._supports_sampling_params():
            req_kwargs["temperature"] = self.temperature
            req_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.tool is not None and not disable_function:
            response = chat_completion_request(
                messages,
                self.tool.functions,
                model=self.model,
                **req_kwargs,
            )
        else:
            response = chat_completion_request(
                messages,
                model=self.model,
                **req_kwargs,
            )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[Message],
        disable_function: bool = False,
        stop: List[str] | str | None = None,
        n: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            disable_function=disable_function,
            stop=stop,
            n=n,
            max_tokens=max_tokens,
            **kwargs,
        )

        full_message = response.choices[0]
        if full_message.finish_reason == "function_call":
            messages.append(full_message["message"])
            func_results = self.tool.call_function(messages, full_message)

            try:
                response = self.chat(messages, disable_function=True)
                return {
                    "response": response,
                    "function_results": func_results,
                }
            except Exception as e:
                print(type(e))
                raise FunctionChatError("Function chat request failed") from e
        else:
            return {
                "response": response,
                "function_results": None,
            }

# Gemini having tools
class GeminiBase:
    def __init__(
        self,
        model: str,
        tool: Iterable[CompletionFunc] | None = None,
        ctx_manager: MoneyManager | None = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.tool = tool
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.enc = tiktoken.get_encoding("cl100k_base")
        if "gemini" in self.model:
            self.max_budget = 64000
        else:
            raise NotImplementedError()
        self.desired_output_length = desired_output_length
        self.temperature = temperature
        # keep as float; Local models may use it, OpenAI Chat models ignore it
        self.repetition_penalty = float(repetition_penalty)

    def _supports_sampling_params(self) -> bool:
        name = self.model
        return not ("gpt-5" in name or any(x in name for x in ("o1", "o3", "o4")))

    def cutoff(self, message: str, budget: int) -> str:
        tokens = self.enc.encode(message)
        if len(tokens) > budget:
            message = self.enc.decode(tokens[:budget])
        return message

    def manage_length(self, messages: List[Message]) -> None:
        last_message = messages[-1]["content"]
        if len(messages) > 1:
            previous_tokens_length = 0
            for msg in messages[:-1]:
                if "content" in msg.keys() and msg["content"] is not None:
                    previous_tokens_length += len(
                        self.enc.encode(msg["content"])
                    )
                elif (
                    "function_call" in msg.keys()
                    and msg["function_call"] is not None
                ):
                    previous_tokens_length += (
                        len(self.enc.encode(msg["function_call"]["arguments"]))
                        + 30
                    )
        else:
            previous_tokens_length = 0
        budget = (
            self.max_budget
            - self.desired_output_length
            - previous_tokens_length
        )
        messages[-1]["content"] = self.cutoff(last_message, budget)

    def chat(
        self,
        messages: List[Message],
        disable_function: bool = False,
        **kwargs,
    ):
        self.manage_length(messages)
        req_kwargs = dict(kwargs)
        if self._supports_sampling_params():
            req_kwargs["temperature"] = self.temperature
            req_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.tool is not None and not disable_function:
            response = chat_completion_request_gemini(
                messages,
                self.tool.functions,
                model=self.model,
                **req_kwargs,
            )
        else:
            response = chat_completion_request_gemini(
                messages,
                model=self.model,
                **req_kwargs,
            )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[Message],
        disable_function: bool = False,
        stop: List[str] | str | None = None,
        n: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            disable_function=disable_function,
            stop=stop,
            n=n,
            max_tokens=max_tokens,
            **kwargs,
        )

        full_message = response.choices[0]
        if full_message.finish_reason == "function_call":
            messages.append(full_message["message"])
            func_results = self.tool.call_function(messages, full_message)

            try:
                response = self.chat(messages, disable_function=True)
                return {
                    "response": response,
                    "function_results": func_results,
                }
            except Exception as e:
                print(type(e))
                raise FunctionChatError("Function chat request failed") from e
        else:
            return {
                "response": response,
                "function_results": None,
            }

# Claude having tools
class ClaudeBase:
    def __init__(
        self,
        model: str,
        tool: Iterable[CompletionFunc] | None = None,
        ctx_manager: MoneyManager | None = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.tool = tool
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.enc = tiktoken.get_encoding("cl100k_base")
        if "claude" in self.model:
            self.max_budget = 128000
        else:
            raise NotImplementedError()
        self.desired_output_length = desired_output_length
        self.temperature = temperature
        self.repetition_penalty = float(repetition_penalty)

    def _supports_sampling_params(self) -> bool:
        name = self.model
        return not ("gpt-5" in name or any(x in name for x in ("o1", "o3", "o4")))

    def cutoff(self, message: str, budget: int) -> str:
        tokens = self.enc.encode(message)
        if len(tokens) > budget:
            message = self.enc.decode(tokens[:budget])
        return message

    def manage_length(self, messages: List[Message]) -> None:
        last_message = messages[-1]["content"]
        if len(messages) > 1:
            previous_tokens_length = 0
            for msg in messages[:-1]:
                if "content" in msg.keys() and msg["content"] is not None:
                    previous_tokens_length += len(
                        self.enc.encode(msg["content"])
                    )
                elif (
                    "function_call" in msg.keys()
                    and msg["function_call"] is not None
                ):
                    previous_tokens_length += (
                        len(self.enc.encode(msg["function_call"]["arguments"]))
                        + 30
                    )
        else:
            previous_tokens_length = 0
        budget = (
            self.max_budget
            - self.desired_output_length
            - previous_tokens_length
        )
        messages[-1]["content"] = self.cutoff(last_message, budget)

    def chat(
        self,
        messages: List[Message],
        disable_function: bool = False,
        **kwargs,
    ):
        self.manage_length(messages)
        req_kwargs = dict(kwargs)
        if self._supports_sampling_params():
            req_kwargs["temperature"] = self.temperature
            req_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.tool is not None and not disable_function:
            response = chat_completion_request(
                messages,
                self.tool.functions,
                model=self.model,
                **req_kwargs,
            )
        else:
            response = chat_completion_request(
                messages,
                model=self.model,
                **req_kwargs,
            )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[Message],
        disable_function: bool = False,
        stop: List[str] | str | None = None,
        n: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            disable_function=disable_function,
            stop=stop,
            n=n,
            max_tokens=max_tokens,
            **kwargs,
        )

        full_message = response.choices[0]
        if full_message.finish_reason == "function_call":
            messages.append(full_message["message"])
            func_results = self.tool.call_function(messages, full_message)

            try:
                response = self.chat(messages, disable_function=True)
                return {
                    "response": response,
                    "function_results": func_results,
                }
            except Exception as e:
                print(type(e))
                raise FunctionChatError("Function chat request failed") from e
        else:
            return {
                "response": response,
                "function_results": None,
            }

# Claude Models
# class ClaudeBase:
#     def __init__(
#         self,
#         model: str,
#         ctx_manager: MoneyManager = None,
#         desired_output_length: int = 512,
#         temperature: float = 1.0,
#     ):
#         self.model = model
#         assert ctx_manager is not None
#         self.ctx_manager = ctx_manager
#         self.enc = tiktoken.get_encoding("cl100k_base")
#         self.max_budget = 128000
#         self.desired_output_length = desired_output_length
#         self.temperature = temperature

#     def cutoff(self, message: Message, budget: int) -> str:
#         tokens = self.enc.encode(message)
#         if len(tokens) > budget:
#             message = self.enc.decode(tokens[:budget])
#         return message

#     def manage_length(self, messages: List[Message]) -> None:
#         last_message = messages[-1]["content"]
#         if len(messages) > 1:
#             previous_tokens_length = 0
#             for msg in messages[:-1]:
#                 if "content" in msg.keys() and msg["content"] is not None:
#                     previous_tokens_length += len(
#                         self.enc.encode(msg["content"])
#                     )
#                 elif (
#                     "function_call" in msg.keys()
#                     and msg["function_call"] is not None
#                 ):
#                     previous_tokens_length += (
#                         len(self.enc.encode(msg["function_call"]["arguments"]))
#                         + 30
#                     )
#         else:
#             previous_tokens_length = 0
#         budget = (
#             self.max_budget
#             - self.desired_output_length
#             - previous_tokens_length
#         )
#         messages[-1]["content"] = self.cutoff(last_message, budget)

#     def chat(self, messages: List[MessageParam], **kwargs):
#         # self.manage_length(messages) # Not needed
#         response = anthropic_chat_completion_request(
#             messages, model=self.model, temperature=self.temperature, **kwargs
#         )
#         self.ctx_manager(response)
#         return response

#     def __call__(
#         self,
#         messages: List[MessageParam],
#         disable_function: bool = False,
#         stop: List[str] | str | None = None,
#         n: int = 1,
#         max_tokens: int | None = None,
#     ):
#         response = self.chat(
#             messages,
#             disable_function=disable_function,
#             stop=stop,
#             n=n,
#             max_tokens=max_tokens,
#         )
#         return {
#             "response": response,
#             "function_results": None,
#         }


# Llama2 Model Base
class LocalBase:
    def __init__(
        self,
        model,
        tool=None,
        ctx_manager: MoneyManager | None = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        instruct_tuned: bool = True,
    ):
        self.model = model
        self.tool = tool
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_budget = 4096
        self.output_budget = 1024
        self.desired_output_length = desired_output_length
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.tok = AutoTokenizer.from_pretrained(self.model)
        self.is_chat_apply = instruct_tuned  # 'instruct' in self.model.lower()

        # 현재 가능한 모든 모델과 model_info 불러오기.
        info_dir = "./loaded_model_info"
        self.model_port_mapping = {}

        # Read all txt files in the directory
        for filename in os.listdir(info_dir):
            if filename.endswith(".txt"):
                port = filename.split(".")[0]
                with open(os.path.join(info_dir, filename), "r", encoding="utf-8") as file:
                    model = file.readline().strip()
                    self.model_port_mapping[model] = port

    def cutoff(self, message: str, budget: int) -> str:
        tokens = self.enc.encode(message)
        if len(tokens) > budget:
            message = self.enc.decode(tokens[:budget])
        return message

    def manage_length(self, messages: List[Message]) -> None:
        last_message = messages[-1]["content"]
        if len(messages) > 1:
            previous_tokens_length = 0
            for msg in messages[:-1]:
                if "content" in msg.keys() and msg["content"] is not None:
                    previous_tokens_length += len(
                        self.enc.encode(msg["content"])
                    )
        else:
            previous_tokens_length = 0
        budget = (
            self.max_budget
            - self.desired_output_length
            - previous_tokens_length
        )
        messages[-1]["content"] = self.cutoff(last_message, budget)

    def chat(
        self, messages: List[Message], lora=None, **kwargs
    ) -> Stream[Completion] | None:
        self.manage_length(messages)
        # Route before call
        openai_api_key = "EMPTY"
        openai_api_base = (
            f"http://localhost:{self.model_port_mapping[self.model]}/v1"
        )
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        # turn messages to prompt
        if self.is_chat_apply:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if messages[-1]["role"] == "assistant":
                key = messages[-1]["content"]
                matches = list(re.finditer(key, prompt))
                if matches:
                    last_match = matches[-1]
                    prompt = prompt[: last_match.end()]
        else:
            prompt = ""
            for message in messages:
                if message["role"] == "system":
                    _prompt = f'### SYSTEM\n{message["content"]}'
                elif message["role"] == "user":
                    _prompt = f'### USER\n{message["content"]}'
                else:
                    _prompt = f'### ASSISTANT\n{message["content"]}'
                prompt += _prompt
            if messages[-1]["role"] != "assistant":
                prompt += "### ASSISTANT\n"

        response = completion_request(
            prompt,
            model=self.model if lora is None else lora,
            temperature=self.temperature,
            client=client,
            **kwargs,
        )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[Message],
        disable_function: bool = False,
        stop: tuple[str, ...] = (
            "### USER",
            "### ASSISTANT",
            "### SYSTEM",
            "###",
            "#",
        ),
        n: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if "gemma" in self.model:  # Gemma model do not have the system prompt!
            system_message = {"role": "system", "content": ""}
            if messages[0]["role"] == "system":
                system_message = messages[0]
                messages = messages[1:]
            for message in messages:
                if message["role"] == "user":
                    message[
                        "content"
                    ] = f'{system_message["content"]}\n{message["content"]}'

        if "SmolLM" in self.model:
            new_messages = []
            if messages[0]["role"] == "system":
                new_user_message = (
                    "\n\n".join(
                        [message["content"] for message in messages[1:-2]]
                    )
                    if messages[-1]["role"] == "assistant"
                    else "\n\n".join(
                        [message["content"] for message in messages[1:-1]]
                    )
                )
                new_messages.append(messages[0])
            else:
                new_user_message = (
                    "\n\n".join(
                        [message["content"] for message in messages[:-2]]
                    )
                    if messages[-1]["role"] == "assistant"
                    else "\n\n".join(
                        [message["content"] for message in messages[:-1]]
                    )
                )
            if messages[-1]["role"] == "assistant":
                new_messages.append(
                    {
                        "role": "user",
                        "content": f"{new_user_message}\n\n{messages[-2]['content']}",
                    }
                )
                new_messages.append(messages[-1])
            else:
                new_messages.append(
                    {
                        "role": "user",
                        "content": f"{new_user_message}\n\n{messages[-1]['content']}",
                    }
                )
            messages = new_messages

        # turn messages to prompt
        if self.is_chat_apply:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if messages[-1]["role"] == "assistant":
                key = messages[-1]["content"]
                matches = list(re.finditer(key, prompt))
                if matches:
                    last_match = matches[-1]
                    prompt = prompt[: last_match.end()]
        else:
            prompt = ""
            for message in messages:
                if message["role"] == "system":
                    _prompt = f'### SYSTEM\n{message["content"]}'
                elif message["role"] == "user":
                    _prompt = f'### USER\n{message["content"]}'
                else:
                    _prompt = f'### ASSISTANT\n{message["content"]}'
                prompt += _prompt
            if messages[-1]["role"] != "assistant":
                prompt += "### ASSISTANT\n"
        desired_output_length = min(
            self.desired_output_length,
            self.max_budget - len(self.enc.encode(prompt)),
        )  #  - 516
        # print(desired_output_length, self.max_budget - len(self.enc.encode(prompt))) # if max_tokens is None else max_tokens
        response = self.chat(
            messages,
            disable_function=disable_function,
            stop=stop,
            n=n,
            max_tokens=desired_output_length,
            repetition_penalty=self.repetition_penalty,
            **kwargs,
        )
        choices = []
        # print(response)
        for choice in response.choices:
            choices.append(
                addict.Dict(
                    {
                        "message": addict.Dict(
                            {
                                "content": choice.text,
                                "role": "assistant",
                            }
                        ),
                        "finish_reason": choice.finish_reason,
                        "index": choice.index,
                        "logprobs": choice.logprobs,
                        "stop_reason": choice.stop_reason,
                    }
                )
            )
        return_response = addict.Dict(
            {
                "id": response.id,
                "choices": choices,
                "model": response.model,
                "object": response.object,
            }
        )
        return {
            "response": return_response,
            "function_results": None,
        }


class Retriever:
    def __init__(self, model: str = "dpr"):
        self.model = model
        self.url = "http://0.0.0.0:9998"
        self.headers = {"Content-Type": "application/json"}

    def retrieve_top_summaries(
        self,
        question: str,
        summaries: List[str],
        encoded_summaries: np.ndarray | None = None,
        topk: int = 5,
    ):
        _ = encoded_summaries  # suppress unused warning
        data = {"question": question, "summaries": summaries, "topk": topk}

        response = requests.post(
            f"{self.url}/{self.model}", headers=self.headers, json=data, timeout=10
        )
        if response.status_code == 200:
            return response.json()["top_summaries"]
        else:
            print("Retrieval Fail... Returns the black response.")
            return []

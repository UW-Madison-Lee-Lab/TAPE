from __future__ import annotations

import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import addict
import numpy as np
import requests
import tiktoken
from openai import OpenAI, Stream
from openai.types import Completion
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random, wait_random_exponential
from termcolor import colored
from transformers import AutoTokenizer

try:
    from anthropic import Anthropic
except Exception as exc:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_IMPORT_ERROR = exc
else:
    _ANTHROPIC_IMPORT_ERROR = None


class _UnavailableClient:
    def __init__(self, name: str, reason: Exception):
        self._name = name
        self._reason = reason

    def __getattr__(self, _attr: str):
        raise RuntimeError(
            f"{self._name} client is unavailable. Original error: {self._reason}"
        ) from self._reason


_LLM_FILE = Path(__file__).resolve()
_SRC_ROOT = _LLM_FILE.parents[1]
_REPO_ROOT = _LLM_FILE.parents[2]


def _candidate_key_paths(key_path: os.PathLike | str) -> List[Path]:
    raw = Path(key_path).expanduser()
    candidates: List[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    if raw.is_absolute():
        _push(raw)
        return candidates

    search_roots = [Path.cwd(), _REPO_ROOT, _SRC_ROOT]
    _push(raw)
    for root in search_roots:
        _push(root / raw)

    # Convenience fallback for passing only file name like "openai-key.env".
    if raw.parent == Path("."):
        for root in search_roots:
            _push(root / "keys" / raw.name)
    else:
        # Also try basename directly under keys/.
        for root in search_roots:
            _push(root / "keys" / raw.name)
    return candidates


def _resolve_key_path(key_path: os.PathLike | str) -> Path:
    candidates = _candidate_key_paths(key_path)
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    candidate_text = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find key file for '{key_path}'. Tried:\n{candidate_text}"
    )


def _normalize_key_line(line: str) -> str:
    value = line.strip()
    if not value or value.startswith("#"):
        return ""
    # Support lines like `OPENAI_API_KEY=...` in addition to plain values.
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", value):
        value = value.split("=", 1)[1].strip()
    return value.strip().strip('"').strip("'")


def _read_key_lines(key_path: os.PathLike | str) -> List[str]:
    resolved = _resolve_key_path(key_path)
    with open(resolved, "r", encoding="utf-8") as f:
        lines = [_normalize_key_line(line) for line in f]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"No API key found in {resolved}.")
    return lines


def setup_openai(key_path: os.PathLike | str = "keys/openai-key.env") -> Dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    organization_key = os.getenv("OPENAI_ORGANIZATION", "").strip()
    if not api_key:
        key_list = _read_key_lines(key_path)
        api_key = key_list[0]
        os.environ["OPENAI_API_KEY"] = api_key
        if len(key_list) > 1 and not organization_key:
            organization_key = key_list[1]
            os.environ["OPENAI_ORGANIZATION"] = organization_key
    config: Dict[str, str] = {"api_key": api_key}
    if organization_key:
        config["organization"] = organization_key
    return config


def setup_gemini(key_path: os.PathLike | str = "keys/gemini-key.env") -> Dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    organization_key = os.getenv("OPENAI_ORGANIZATION", "").strip()
    if not api_key:
        key_list = _read_key_lines(key_path)
        api_key = key_list[0]
        if len(key_list) > 1 and not organization_key:
            organization_key = key_list[1]
    os.environ["OPENAI_API_KEY"] = api_key
    if organization_key:
        os.environ["OPENAI_ORGANIZATION"] = organization_key
    config: Dict[str, str] = {
        "api_key": api_key,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    if organization_key:
        config["organization"] = organization_key
    return config


def setup_anthropic(
    key_path: os.PathLike | str = "keys/anthropic-key.env",
) -> Dict[str, str]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        key_list = _read_key_lines(key_path)
        api_key = key_list[0]
    os.environ["ANTHROPIC_API_KEY"] = api_key
    return {"api_key": api_key}


def _safe_build_client(name: str, factory):
    try:
        return factory()
    except Exception as exc:
        return _UnavailableClient(name, exc)


client = _safe_build_client("openai", lambda: OpenAI(**setup_openai()))
gemini_client = _safe_build_client("gemini", lambda: OpenAI(**setup_gemini()))

if Anthropic is None:
    anthropic_client = _UnavailableClient("anthropic", _ANTHROPIC_IMPORT_ERROR)  # type: ignore[arg-type]
else:
    anthropic_client = _safe_build_client(
        "anthropic", lambda: Anthropic(**setup_anthropic())
    )


def pretty_print_conversation(messages: List[Dict[str, Any]]) -> None:
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(
                f"function ({message['name']}): {message['content']}\n"
            )

    for idx, formatted_message in enumerate(formatted_messages):
        role = messages[idx]["role"]
        print(colored(formatted_message, role_to_color[role]))


class MoneyManager:
    def __init__(self, model: str = "gpt-3.5-turbo-0613"):
        self.total_cost = 0.0
        self.model = model
        if self.model == "gpt-3.5-turbo-16k-0613":
            self.input_cost = 0.003
            self.output_cost = 0.004
        elif self.model == "gpt-3.5-turbo-0613":
            self.input_cost = 0.0015
            self.output_cost = 0.002
        elif self.model == "gpt-3.5-turbo-1106":
            self.input_cost = 0.001
            self.output_cost = 0.002
        elif self.model == "gpt-3.5-turbo":
            self.input_cost = 0.001
            self.output_cost = 0.002
        elif self.model == "gpt-4-turbo-preview":
            self.input_cost = 0.01
            self.output_cost = 0.03
        elif self.model == "gpt-4-turbo":
            self.input_cost = 0.01
            self.output_cost = 0.03
        elif self.model == "gpt-4-1106-preview":
            self.input_cost = 0.01
            self.output_cost = 0.03
        elif self.model == "gpt-4":
            self.input_cost = 0.03
            self.output_cost = 0.06
        elif self.model == "text-embedding-ada-002":
            self.input_cost = 0.0001
            self.output_cost = 0.0
        elif self.model == "claude-3-opus-20240229":
            self.input_cost = 0.015
            self.output_cost = 0.075
        elif self.model == "claude-3-opus-20240229":
            self.input_cost = 0.003
            self.output_cost = 0.015
        elif self.model == "gpt-4o":
            self.input_cost = 0.0025
            self.output_cost = 0.01
        elif self.model == "gpt-4o-mini":
            self.input_cost = 0.15 / 1000
            self.output_cost = 0.6 / 1000
        elif self.model == "gpt-4o-2024-08-06":
            self.input_cost = 2.5 / 1000
            self.output_cost = 10 / 1000
        elif self.model == "o1-preview":
            self.input_cost = 15 / 1000
            self.output_cost = 60 / 1000
        elif self.model == "o1-mini":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "o1-pro":
            self.input_cost = 15 / 100
            self.output_cost = 60 / 100
        elif self.model == "o3":
            self.input_cost = 2 / 1000
            self.output_cost = 8 / 1000
        elif self.model == "o3-pro":
            self.input_cost = 20 / 1000
            self.output_cost = 80 / 1000
        elif self.model == "o3-mini":
            self.input_cost = 20 / 1000
            self.output_cost = 80 / 1000
        elif self.model == "o4-mini":
            self.input_cost = 1.1 / 1000
            self.output_cost = 4.4 / 1000
        elif self.model == "gpt-4.1":
            self.input_cost = 0.002
            self.output_cost = 0.008
        elif self.model == "gpt-4.1-mini":
            self.input_cost = 0.0004
            self.output_cost = 0.0016
        elif self.model == "gpt-4.1-nano":
            self.input_cost = 0.0001
            self.output_cost = 0.0004
        elif self.model == "gpt-5":
            self.input_cost = 0.00125
            self.output_cost = 0.01
        elif self.model == "gpt-5-mini":
            self.input_cost = 0.00025
            self.output_cost = 0.002
        elif self.model == "gpt-5-nano":
            self.input_cost = 0.00005
            self.output_cost = 0.0004
        else:
            self.input_cost = 0.0
            self.output_cost = 0.0

    def __call__(self, response: Any = None) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            print("No usage in response")
            print(response)
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        input_cost = prompt_tokens / 1000 * self.input_cost

        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is not None:
            output_cost = completion_tokens / 1000 * self.output_cost
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details is not None:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
                if reasoning_tokens is not None:
                    output_cost += reasoning_tokens / 1000 * self.output_cost
                rejected_tokens = getattr(
                    completion_details, "rejected_prediction_tokens", None
                )
                if rejected_tokens is not None:
                    output_cost += rejected_tokens / 1000 * self.output_cost
        else:
            output_cost = 0.0

        self.total_cost += input_cost + output_cost

    def refresh(self) -> None:
        self.total_cost = 0.0


class Logger:
    def __init__(
        self,
        agent_name: str = "sql",
        log_root_path: os.PathLike | str = "logs/custom/",
    ):
        nowdate = datetime.now().strftime("%Y%m%d")
        self.log_path = Path(log_root_path) / agent_name / nowdate
        os.makedirs(self.log_path, exist_ok=True)

    def __call__(self, messages: List[Dict[str, Any]]):
        nowtime = datetime.now().time().strftime("%H%M%S")
        filepath = self.log_path / (nowtime + ".json")
        with open(filepath, "w+", encoding="utf-8") as f:
            json.dump(
                messages,
                f,
                ensure_ascii=False,
                indent=4,
            )



CompletionFunc = Any
Message = Dict[str, Any]


@retry(wait=wait_random(min=1, max=10), stop=stop_after_attempt(5))
def chat_completion_request(
    messages: List[Message],
    functions: Iterable[CompletionFunc] | None = None,
    function_call: Any | None = None,
    model: str = "gpt-3.5-turbo-0613",
    client: OpenAI = client,
    **kwargs,
) -> Stream[ChatCompletion] | None:
    json_data: Dict[str, Any] = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})

    if "stop" in kwargs and kwargs["stop"] is not None and "gpt-5" not in model:
        json_data.update({"stop": kwargs["stop"]})
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        json_data.update({"temperature": kwargs["temperature"]})
    if "n" in kwargs and kwargs["n"] is not None:
        json_data.update({"n": kwargs["n"]})
    if "max_tokens" in kwargs and kwargs["max_tokens"] is not None and "gpt-5" not in model:
        json_data.update({"max_tokens": kwargs["max_tokens"]})
    if "json_mode" in kwargs and kwargs["json_mode"]:
        json_data.update({"response_format": {"type": "json_object"}})

    if "verbosity" in kwargs and kwargs["verbosity"] is not None:
        json_data.update({"verbosity": kwargs["verbosity"]})
    elif "gpt-5" in model:
        json_data.update({"verbosity": "medium"})

    if "reasoning_effort" in kwargs and kwargs["reasoning_effort"] is not None:
        json_data.update({"reasoning_effort": kwargs["reasoning_effort"]})

    if "response_format" in kwargs and kwargs["response_format"] is not None:
        json_data.update({"response_format": kwargs["response_format"]})
        try:
            return client.beta.chat.completions.parse(**json_data)
        except Exception:
            return client.chat.completions.create(**json_data)

    return client.chat.completions.create(**json_data)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_request(
    prompt: str,
    model: str = "gpt-3.5-turbo-0613",
    client: OpenAI = client,
    **kwargs,
) -> Completion | None:
    json_data: Dict[str, Any] = {"model": model, "prompt": prompt}
    extra_data: Dict[str, Any] = {}

    args_keys = ["stop", "temperature", "n", "max_tokens", "top_k", "top_p", "do_sample"]
    extra_keys = ["repetition_penalty"]

    for args_key in args_keys:
        if args_key in kwargs and kwargs[args_key] is not None:
            json_data.update({args_key: kwargs[args_key]})

    for args_key in extra_keys:
        if args_key in kwargs and kwargs[args_key] is not None:
            extra_data.update({args_key: kwargs[args_key]})

    if extra_data:
        json_data.update({"extra_body": extra_data})

    return client.completions.create(**json_data)


def chat_completion_request_gemini(
    messages: List[Message],
    functions: Iterable[CompletionFunc] | None = None,
    function_call: Any | None = None,
    model: str = "gemini-3-flash-preview",
    client: OpenAI = gemini_client,
    **kwargs,
) -> Stream[ChatCompletion] | None:
    return chat_completion_request(
        messages=messages,
        functions=functions,
        function_call=function_call,
        model=model,
        client=client,
        **kwargs,
    )


def completion_request_gemini(
    prompt: str,
    model: str = "gemini-3-flash-preview",
    client: OpenAI = gemini_client,
    **kwargs,
) -> Completion | None:
    return completion_request(prompt=prompt, model=model, client=client, **kwargs)


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

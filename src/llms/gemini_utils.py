import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from openai import OpenAI, Stream
from openai.types import Completion, Embedding
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random,
    wait_random_exponential,
)
from termcolor import colored

from .typing import (
    CompletionFunc,
    CompletionFuncCall,
    Message,
)

def setup_openai(
    key_path: os.PathLike = "keys/gemini-key.env",
) -> Dict[str, str]:
    with open(key_path, "r") as f:
        key_list = f.readlines()

    if len(key_list) > 1:
        api_key = key_list[0].strip()
        organization_key = key_list[1].strip()
        os.environ["OPENAI_ORGANIZATION"] = organization_key
        os.environ["OPENAI_API_KEY"] = api_key
        return {"organization": organization_key, "api_key": api_key}
    else:
        api_key = key_list[0].strip()
        os.environ["OPENAI_API_KEY"] = api_key
        return {"api_key": api_key, "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"}

client = OpenAI(**setup_openai())


@retry(wait=wait_random(min=1, max=10), stop=stop_after_attempt(5))
def chat_completion_request(
    messages: List[Message],
    functions: Iterable[CompletionFunc] | None = None,
    function_call: CompletionFuncCall | None = None,
    model: str = "gemini-3-flash-preview",
    client: OpenAI = client,
    **kwargs,
) -> Stream[ChatCompletion] | None:
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})

    if "stop" in kwargs.keys() and kwargs["stop"] is not None:
        if "gpt-5" not in model:
            json_data.update({"stop": kwargs["stop"]})
    if "temperature" in kwargs.keys() and kwargs["temperature"] is not None:
        json_data.update({"temperature": kwargs["temperature"]})
    if "n" in kwargs.keys() and kwargs["n"] is not None:
        json_data.update({"n": kwargs["n"]})
    if "max_tokens" in kwargs.keys() and kwargs["max_tokens"] is not None:
        if "gpt-5" not in model:
            json_data.update({"max_tokens": kwargs["max_tokens"]})
    if "json_mode" in kwargs.keys() and kwargs["json_mode"] is not None:
        if kwargs["json_mode"]:
            json_data.update({"response_format": {"type": "json_object"}})
    # json_data.update({"request_timeout": 300})
    if "verbosity" in kwargs.keys() and kwargs["verbosity"] is not None:
        assert kwargs["verbosity"] in ["low", "medium", "high"]
        json_data.update({"verbosity": kwargs["verbosity"]})
    elif "gpt-5" in model:
        json_data.update({"verbosity": "medium"})

    if "reasoning_effort" in kwargs.keys() and kwargs["reasoning_effort"] is not None:
        assert kwargs["reasoning_effort"] in ["minimal", "low", "medium", "high"]
        json_data.update({"reasoning_effort": kwargs["reasoning_effort"]})
    elif "gpt-5" in model:
        json_data.update({"verbosity": "medium"})

    if (
        "response_format" in kwargs.keys()
        and kwargs["response_format"] is not None
    ):
        json_data.update({"response_format": kwargs["response_format"]})
        # if the response_format is given, we need to use the "beta" call function
        try:
            response = client.beta.chat.completions.parse(**json_data)
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            raise e
    else:
        try:
            response = client.chat.completions.create(**json_data)
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            raise e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_request(
    prompt, model: str = "gemini-3-flash-preview", client: OpenAI = client, **kwargs
) -> ChatCompletion | None:
    json_data = {"model": model, "prompt": prompt}
    extra_data = {}

    args_keys = [
        "stop",
        "temperature",
        "n",
        "max_tokens",
        "top_k",
        "top_p",
        "do_sample",
    ]
    extra_keys = ["repetition_penalty"]

    for args_key in args_keys:
        if args_key in kwargs.keys() and kwargs[args_key] is not None:
            json_data.update({args_key: kwargs[args_key]})

    for args_key in extra_keys:
        if args_key in kwargs.keys() and kwargs[args_key] is not None:
            extra_data.update({args_key: kwargs[args_key]})

    if len(extra_data) > 0:
        json_data.update({"extra_body": extra_data})

    # json_data.update({"request_timeout": 30})
    try:
        response = client.completions.create(**json_data)
        return response
    except Exception as e:
        print("Unable to generate Completion response")
        print(f"Exception: {e}")
        raise e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def embedding_request(
    text: str, model: str = "text-embedding-3-small"
) -> Embedding | None:
    text = text.replace("\n", " ")

    try:
        return (
            client.embeddings.create(input=[text], model=model)
            .data[0]
            .embedding
        )
    except Exception as e:
        print("Unable to generate Embedding response")
        print(f"Exception: {e}")
        raise e


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
        elif self.model == 'gpt-4.1':
            self.input_cost = 0.002
            self.output_cost = 0.008
        elif self.model == 'gpt-4.1-mini':
            self.input_cost = 0.0004
            self.output_cost = 0.0016
        elif self.model == 'gpt-4.1-nano':
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
            # raise NotImplementedError

    def __call__(self, response: Completion | None = None) -> None:
        if response.usage is None:
            print("No usage in response")
            print(response)
            return
        input_cost = response.usage.prompt_tokens / 1000 * self.input_cost
        
        if (
            response.usage.completion_tokens is not None
        ):  # "completion_tokens" in response.usage.keys():
            output_cost = (
                response.usage.completion_tokens / 1000 * self.output_cost
            )
            
            if response.usage.completion_tokens_details.reasoning_tokens is not None:
                reasoning_cost = (
                    response.usage.completion_tokens_details.reasoning_tokens
                    / 1000
                    * self.output_cost
                )
                output_cost += reasoning_cost
            
            if response.usage.completion_tokens_details.rejected_prediction_tokens is not None:
                rejected_cost = (
                    response.usage.completion_tokens_details.rejected_prediction_tokens
                    / 1000
                    * self.output_cost
                )
                output_cost += rejected_cost

        else:
            output_cost = 0.0
        self.total_cost += input_cost + output_cost

    def refresh(self) -> None:
        self.total_cost = 0.0


class Logger:
    def __init__(
        self,
        agent_name: str = "sql",
        log_root_path: os.PathLike = "logs/custom/",
    ):
        nowdate = datetime.now().strftime("%Y%m%d")

        self.log_path = Path(log_root_path) / agent_name / nowdate
        os.makedirs(self.log_path, exist_ok=True)

    def __call__(self, messages: List[Message]):
        nowtime = datetime.now().time().strftime("%H%M%S")
        filepath = self.log_path / (nowtime + ".json")
        with open(filepath, "w+", encoding="utf-8") as f:
            json.dump(
                messages,
                f,
                ensure_ascii=False,
                indent=4,
            )
        # print(f"Dump output complete {str(filepath)}")


def pretty_print_conversation(messages: List[Message]):
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
            formatted_messages.append(
                f"assistant: {message['function_call']}\n"
            )
        elif message["role"] == "assistant" and not message.get(
            "function_call"
        ):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(
                f"function ({message['name']}): {message['content']}\n"
            )
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[
                    messages[formatted_messages.index(formatted_message)][
                        "role"
                    ]
                ],
            )
        )


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Say hi."},
        {"role": "user", "content": "Hello."},
    ]
    message = chat_completion_request(
        messages, model="gpt-4-turbo-preview", max_tokens=100, temperature=0.5
    )
    print(message)
    import pdb

    pdb.set_trace()

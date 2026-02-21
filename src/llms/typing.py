from typing import Any, Dict, TypeAlias

from openai.types.chat import completion_create_params

CompletionFunc: TypeAlias = completion_create_params.Function

CompletionFuncCall: TypeAlias = completion_create_params.FunctionCall

Message: TypeAlias = Dict[str, Any]
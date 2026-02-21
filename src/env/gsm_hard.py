from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import inspect
import json
import os
from pathlib import Path

try:
    from tools import arithmetic_tools
    from tools.base import Tool
except ModuleNotFoundError as exc:
    if exc.name not in {"tools", "tools.base"}:
        raise
    from src.tools import arithmetic_tools
    from src.tools.base import Tool


class GsmHardEnv:
    def __init__(self, data_path: Optional[str] = None, deterministic: bool = True):
        if data_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            data_path = str(repo_root / "data" / "gsm_hard_train_processed_sampled.json")
        self.data_path = data_path
        self.deterministic = deterministic
        self.examples = self._load_examples(data_path)
        self._tool_registry, self._max_tool_cost = self._build_tool_registry()
        self.current_index: Optional[int] = None
        self.current_example: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    def _load_examples(self, data_path: str) -> List[Dict[str, Any]]:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = data.get("examples")
        if not isinstance(examples, list):
            raise ValueError("Expected 'examples' list in dataset.")
        return examples

    def _build_tool_registry(self) -> Tuple[Dict[str, Tool], float]:
        instances_by_class: Dict[type, Tool] = {}
        registry: Dict[str, Tool] = {}

        for name in getattr(arithmetic_tools, "__all__", []):
            cls = getattr(arithmetic_tools, name, None)
            if cls is None or not inspect.isclass(cls) or not issubclass(cls, Tool):
                continue
            if cls.__module__ != arithmetic_tools.__name__:
                continue
            inst = cls()
            instances_by_class[cls] = inst
            registry[inst.name] = inst
            registry[cls.__name__] = inst

        for attr in dir(arithmetic_tools):
            if attr.startswith("_"):
                continue
            value = getattr(arithmetic_tools, attr)
            if (
                inspect.isclass(value)
                and issubclass(value, Tool)
                and value.__module__ == arithmetic_tools.__name__
            ):
                inst = instances_by_class.get(value)
                if inst is None:
                    inst = value()
                    instances_by_class[value] = inst
                registry.setdefault(attr, inst)

        max_cost = 0.0
        for inst in registry.values():
            max_cost = max(max_cost, self._get_tool_cost(inst, success=False))
        return registry, max_cost

    def reset(self, index: int) -> str:
        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if index < 0 or index >= len(self.examples):
            raise IndexError("index out of range.")
        self.current_index = index
        self.current_example = self.examples[index]
        self.last_error = None
        return self.current_example.get("question", "")

    def step(
        self,
        function_name: str,
        arguments: Optional[Union[Tuple[Any, ...], List[Any], Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], float]:
        self.last_error = None
        tool = self._tool_registry.get(function_name)
        if tool is None:
            self.last_error = f"Unknown tool: {function_name}"
            return None, self._max_tool_cost
        try:
            if arguments is None:
                result = tool()
            elif isinstance(arguments, dict):
                result = tool(**arguments)
            elif isinstance(arguments, (list, tuple)):
                result = tool(*arguments)
            else:
                result = tool(arguments)
            cost = self._get_tool_cost(tool, success=True)
            return result, cost
        except Exception as exc:
            self.last_error = str(exc)
            cost = self._get_tool_cost(tool, success=False)
            return None, cost

    def _get_tool_cost(self, tool: Tool, success: bool) -> float:
        if not success and hasattr(tool, "cost_max"):
            return float(getattr(tool, "cost_max"))
        if self.deterministic and hasattr(tool, "cost_mu"):
            return float(getattr(tool, "cost_mu"))
        if not self.deterministic:
            sampler = getattr(tool, "default_cost_sampler", None)
            if callable(sampler):
                try:
                    return float(sampler())
                except Exception:
                    pass
        if hasattr(tool, "default_cost"):
            return float(getattr(tool, "default_cost"))
        if hasattr(tool, "cost_mu"):
            return float(getattr(tool, "cost_mu"))
        return 0.0

    def __len__(self) -> int:
        return len(self.examples)

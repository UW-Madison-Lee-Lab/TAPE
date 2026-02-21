"""
Run ReAct / PA / Ours agents on MuSiQue dataset with arithmetic tools,
controlled by time and cost budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.gsm_hard import GsmHardEnv
from real_agents.react_agent import ReACTAgent
from real_agents.pa_agent import PAAgent
from real_agents.ours_graph_ilp import OursGraphILPAgentArithmetic
from real_agents.prompts.arithmetic_prompts import (
    ARITHMETIC_FORCE_FINAL_ANSWER_HINT,
    ARITHMETIC_RESPONSE_FORMAT_HINT,
    build_arithmetic_system_prompt,
    format_arithmetic_tool_descriptions,
)
from utils.llm import pretty_print_conversation
from utils.io import save_csv, save_json, ensure_dir

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore
    JOBLIB_AVAILABLE = False

# ---- Default constraint pairs (time_budget, cost_budget) ----
DEFAULT_TIME_CONSTRAINTS = [0.6, 0.8, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]
DEFAULT_COST_CONSTRAINTS = [0.02, 0.04, 0.02, 0.04, 0.05, 0.07, 0.06, 0.1]


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "?"
    total = int(seconds + 0.5)
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


def _read_episode_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_system_message(env: GsmHardEnv) -> str:
    """Build arithmetic system prompt from prompt module."""
    return build_arithmetic_system_prompt(env._tool_registry)


def _cost_enabled(cost_budget: Optional[float]) -> bool:
    return cost_budget is not None


def _time_enabled(time_budget: Optional[float]) -> bool:
    return time_budget is not None


def _budget_exhausted(
    time_used: float,
    time_budget: Optional[float],
    cost_used: float,
    cost_budget: Optional[float],
) -> bool:
    if _time_enabled(time_budget) and time_used >= float(time_budget):
        return True
    if _cost_enabled(cost_budget) and cost_used >= float(cost_budget):
        return True
    return False


def _constraint_tag(time_budget: Optional[float], cost_budget: Optional[float]) -> str:
    if not _time_enabled(time_budget) and not _cost_enabled(cost_budget):
        return "unbounded"
    if not _time_enabled(time_budget) and _cost_enabled(cost_budget):
        return f"notime_c{float(cost_budget):.4f}"
    if _time_enabled(time_budget) and not _cost_enabled(cost_budget):
        return f"t{float(time_budget):.2f}_nocost"
    return f"t{float(time_budget):.2f}_c{float(cost_budget):.4f}"


def _format_budget_intro(time_budget: Optional[float], cost_budget: Optional[float]) -> str:
    if not _time_enabled(time_budget) and not _cost_enabled(cost_budget):
        return "No time/cost budget constraints."
    if _time_enabled(time_budget) and not _cost_enabled(cost_budget):
        return (
            f"Time budget: {float(time_budget):.3f}s\n"
            f"Time remaining: {float(time_budget):.3f}s"
        )
    if not _time_enabled(time_budget) and _cost_enabled(cost_budget):
        return (
            f"Cost budget: ${float(cost_budget):.4f}\n"
            f"Cost remaining: ${float(cost_budget):.4f}"
        )
    return (
        f"Time budget: {float(time_budget):.3f}s | Cost budget: ${float(cost_budget):.4f}\n"
        f"Time remaining: {float(time_budget):.3f}s | Cost remaining: ${float(cost_budget):.4f}"
    )


def _format_remaining_feedback(
    time_budget: Optional[float],
    time_used: float,
    cost_budget: Optional[float],
    cost_used: float,
) -> str:
    time_remaining = (
        max(0.0, float(time_budget) - time_used) if _time_enabled(time_budget) else None
    )
    cost_remaining = (
        max(0.0, float(cost_budget) - cost_used) if _cost_enabled(cost_budget) else None
    )
    if time_remaining is not None and cost_remaining is not None:
        return (
            f"Time remaining: {time_remaining:.3f}s | "
            f"Cost remaining: ${cost_remaining:.4f}"
        )
    if time_remaining is not None:
        return f"Time remaining: {time_remaining:.3f}s"
    if cost_remaining is not None:
        return f"Cost remaining: ${cost_remaining:.4f}"
    return ""


def _format_step_usage(
    step_time: float,
    step_cost: float,
    time_budget: Optional[float],
    cost_budget: Optional[float],
) -> str:
    parts: List[str] = []
    if _time_enabled(time_budget):
        parts.append(f"time_used: {step_time:.4f}s")
    if _cost_enabled(cost_budget):
        parts.append(f"cost: ${step_cost:.4f}")
    if not parts:
        return ""
    return "(" + ", ".join(parts) + ")"


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _parse_structured_json_action(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON action in strong-conditioning shape:
    {"thought": str, "tool_name": str, "arguments": {...}}
    """
    blob = _extract_json_blob(text)
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _force_react_boxed_answer(
    agent: ReACTAgent,
    messages: List[Dict[str, Any]],
    parse_retries: int,
    print_episode: bool,
) -> str:
    """Force ReAct to terminate with boxed final answer; fallback to Unknown."""
    retries = max(0, parse_retries)
    for attempt in range(retries + 1):
        if attempt > 0:
            messages = agent.observe(messages, ARITHMETIC_FORCE_FINAL_ANSWER_HINT)

        preds = agent.act(messages)
        if print_episode:
            pretty_print_conversation(messages[-2:])
        response_text = preds[0] if preds else ""
        final_ans = _parse_final_answer(response_text)
        if final_ans is not None:
            return final_ans

        if messages and messages[-1].get("role") == "assistant":
            messages.pop()

    messages.append(
        {"role": "assistant", "content": "Action: Answer is \\boxed{Unknown}."}
    )
    return "Unknown"


def _get_tool_time(tool, deterministic: bool, success: bool) -> float:
    """Sample or return deterministic execution time for a tool."""
    if not success and hasattr(tool, "execution_time_max"):
        return float(getattr(tool, "execution_time_max"))
    if deterministic and hasattr(tool, "execution_time_mu"):
        return float(getattr(tool, "execution_time_mu"))
    if not deterministic:
        sampler = getattr(tool, "default_execution_time_sampler", None)
        if callable(sampler):
            try:
                return float(sampler())
            except Exception:
                pass
    if hasattr(tool, "default_execution_time"):
        return float(getattr(tool, "default_execution_time"))
    if hasattr(tool, "execution_time_mu"):
        return float(getattr(tool, "execution_time_mu"))
    return 0.0


def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse a tool call from agent response text.

    Supports formats like:
        Action: tool_name(a=1, b=2)
        Action: tool_name {"a": 1, "b": 2}
        Action: tool_name(1, 2)
    """
    if not text:
        return None

    structured = _parse_structured_json_action(text)
    if isinstance(structured, dict):
        tool_name = structured.get("tool_name")
        arguments = structured.get("arguments", {})
        if isinstance(tool_name, str):
            if tool_name.lower() == "answer":
                return None
            if not isinstance(arguments, dict):
                arguments = {}
            return tool_name.strip(), arguments

    match = re.search(
        r"Action\s*:\s*(\w+)\s*\(([^)]*)\)",
        text,
        re.IGNORECASE,
    )
    if match:
        name = match.group(1).strip()
        if name.lower() == "answer":
            return None
        args_str = match.group(2).strip()
        args = _parse_args(args_str)
        return name, args

    match = re.search(
        r"Action\s*:\s*(\w+)\s*(\{.*\})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        name = match.group(1).strip()
        if name.lower() == "answer":
            return None
        try:
            args = json.loads(match.group(2))
            if isinstance(args, dict):
                return name, args
        except json.JSONDecodeError:
            pass
        return name, {}

    match = re.search(r"Action\s*:\s*(\w+)", text, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        if name.lower() == "answer":
            return None
        return name, {}

    return None


def _parse_args(args_str: str) -> Dict[str, Any]:
    """Parse 'a=1, b=2' or '1, 2' into a dict."""
    if not args_str.strip():
        return {}
    if "=" in args_str:
        result = {}
        for part in args_str.split(","):
            part = part.strip()
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
            result[k] = v
        return result
    # positional args -> use "a", "b" as keys for binary ops, "n" for unary
    parts = [p.strip() for p in args_str.split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(json.loads(p))
        except (json.JSONDecodeError, ValueError):
            vals.append(p)
    if len(vals) == 1:
        return {"n": vals[0]}
    if len(vals) == 2:
        return {"a": vals[0], "b": vals[1]}
    return {f"arg{i}": v for i, v in enumerate(vals)}


def _parse_final_answer(text: str) -> Optional[str]:
    """Parse a final answer from agent response text.

    Supports:
        Action: Answer is \boxed{<answer>}.
        Action: Answer is Unknown.
        Final Answer: <answer>
        Answer: <answer>
    """
    if not text:
        return None

    structured = _parse_structured_json_action(text)
    if isinstance(structured, dict):
        tool_name = structured.get("tool_name")
        arguments = structured.get("arguments", {})
        if isinstance(tool_name, str) and tool_name.lower() == "answer":
            if isinstance(arguments, dict):
                ans = arguments.get("answer", arguments.get("final_answer"))
                if ans is not None:
                    return str(ans).strip().strip("\"'")
            if "answer" in structured:
                return str(structured["answer"]).strip().strip("\"'")
            return "Unknown"
        if "final_answer" in structured:
            ans = structured.get("final_answer")
            if ans is not None:
                return str(ans).strip().strip("\"'")

    boxed_match = re.search(
        r"Action\s*:\s*Answer\s+is\s*\\boxed\s*\{([^{}]+)\}\s*\.?",
        text,
        re.IGNORECASE,
    )
    if boxed_match:
        answer = boxed_match.group(1).strip().strip("\"'")
        return answer if answer else None

    unknown_match = re.search(
        r"Action\s*:\s*Answer\s+is\s*(Unknown)\s*\.?",
        text,
        re.IGNORECASE,
    )
    if unknown_match:
        return "Unknown"

    match = re.search(r"(?:Final\s+)?Answer\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().strip("\"'")
        return answer if answer else None
    return None


def _extract_number(text: str) -> Optional[float]:
    """Try to extract a numeric value from a string."""
    if text is None:
        return None
    text = str(text).strip()
    text = re.sub(r"[,\$%]", "", text)
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _check_answer(predicted: Optional[str], gold: str) -> bool:
    """Check if the predicted answer matches the gold answer.

    Success criteria:
      - Exact string match (after normalization), OR
      - Relative error <= 5% when both are numeric.
    """
    if predicted is None:
        return False
    pred = str(predicted).strip()
    gold = str(gold).strip()
    # exact match
    if pred.lower() == gold.lower():
        return True
    # numeric comparison with 5% relative error tolerance
    pred_f = _extract_number(pred)
    gold_f = _extract_number(gold)
    if pred_f is not None and gold_f is not None:
        if pred_f == gold_f:
            return True
        if gold_f != 0:
            rel = abs(pred_f - gold_f) / abs(gold_f)
        else:
            rel = abs(pred_f - gold_f)
        if rel <= 0.05:
            return True
    return False


def _build_agent(args, agent_type: str):
    if agent_type == "pa":
        return PAAgent(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            task="arithmetic",
        )
    if agent_type == "ours_graph_ilp":
        return OursGraphILPAgentArithmetic(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    return ReACTAgent(
        model=args.model,
        fine_tuned_model=args.fine_tuned_model,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        instruct_tuned=True,
        debug_mode=False,
        is_print=False,
        task="arithmetic",
    )


def _run_episode(
    env: GsmHardEnv,
    agent: Union[ReACTAgent, PAAgent, OursGraphILPAgentArithmetic],
    example_index: int,
    time_budget: Optional[float],
    cost_budget: Optional[float],
    max_steps: int,
    parse_retries: int,
    print_episode: bool = False,
    deterministic: bool = True,
) -> Tuple[bool, int, float, float, Optional[str], str, List[Dict[str, Any]]]:
    """Run a single episode on one MuSiQue example.

    Returns:
        (success, steps, time_used, cost_used, predicted_answer, gold_answer, messages)
    """
    question = env.reset(example_index)
    gold_answer = env.current_example.get("answer", "")
    tool_descriptions = format_arithmetic_tool_descriptions(env._tool_registry)

    if isinstance(agent, OursGraphILPAgentArithmetic):
        agent.configure_tools(env._tool_registry)
        system_msg = agent.generate_initial_message(tool_registry=env._tool_registry)
    elif type(agent) is ReACTAgent:
        system_msg = agent.generate_initial_message(tool_registry=env._tool_registry)
    elif type(agent) is PAAgent:
        planning_observation = (
            f"Question: {question}\n"
            f"{_format_budget_intro(time_budget, cost_budget)}\n"
            "Create a tool-use plan before solving."
        )
        plan_text = agent.generate_plan(
            observation=planning_observation,
            steps_remaining=max_steps,
            transition_info_text=tool_descriptions,
        )
        system_msg = agent.generate_initial_message(
            plan_text=plan_text or "",
            transition_info_text=tool_descriptions,
        )
    else:
        system_msg = _build_system_message(env)

    messages = [{"role": "system", "content": system_msg}]
    agent.is_print = print_episode

    time_used = 0.0
    cost_used = 0.0
    predicted_answer = None

    initial_obs = f"Question: {question}\n{_format_budget_intro(time_budget, cost_budget)}"
    messages = agent.observe(messages, initial_obs)

    for step in range(max_steps):
        preds = agent.act(messages)
        if print_episode:
            pretty_print_conversation(messages[-2:])

        response_text = preds[0] if preds else ""

        final_ans = _parse_final_answer(response_text)
        if final_ans is not None:
            predicted_answer = final_ans
            break

        tool_call = _parse_tool_call(response_text)
        retry = 0
        while tool_call is None and final_ans is None and retry < parse_retries:
            retry += 1
            if messages and messages[-1].get("role") == "assistant":
                messages.pop()
            preds = agent.act(messages)
            if print_episode:
                pretty_print_conversation(messages[-2:])
            response_text = preds[0] if preds else ""
            final_ans = _parse_final_answer(response_text)
            if final_ans is not None:
                predicted_answer = final_ans
                break
            tool_call = _parse_tool_call(response_text)

        if predicted_answer is not None:
            break

        if tool_call is None:
            remaining = _format_remaining_feedback(time_budget, time_used, cost_budget, cost_used)
            obs = (
                f"{ARITHMETIC_RESPONSE_FORMAT_HINT}\n{remaining}"
                if remaining
                else ARITHMETIC_RESPONSE_FORMAT_HINT
            )
            messages = agent.observe(messages, obs)
            continue

        tool_name, tool_args = tool_call
        result, step_cost = env.step(tool_name, tool_args)

        tool_inst = env._tool_registry.get(tool_name)
        step_time = 0.0
        if tool_inst is not None:
            step_time = _get_tool_time(tool_inst, deterministic, result is not None)

        time_used += step_time
        cost_used += step_cost
        if isinstance(agent, OursGraphILPAgentArithmetic):
            agent.record_tool_execution(
                tool_name=tool_name,
                step_time=step_time,
                step_cost=step_cost,
            )

        if result is not None:
            usage = _format_step_usage(step_time, step_cost, time_budget, cost_budget)
            remaining = _format_remaining_feedback(time_budget, time_used, cost_budget, cost_used)
            obs_parts = [f"Tool result: {result}"]
            if usage:
                obs_parts.append(usage)
            if remaining:
                obs_parts.append(remaining)
            obs = "\n".join(obs_parts)
        else:
            error_msg = env.last_error or "Unknown error"
            usage = _format_step_usage(step_time, step_cost, time_budget, cost_budget)
            remaining = _format_remaining_feedback(time_budget, time_used, cost_budget, cost_used)
            obs_parts = [f"Tool error: {error_msg}"]
            if usage:
                obs_parts.append(usage)
            if remaining:
                obs_parts.append(remaining)
            obs = "\n".join(obs_parts)

        if _budget_exhausted(time_used, time_budget, cost_used, cost_budget):
            obs += (
                f"\n{ARITHMETIC_FORCE_FINAL_ANSWER_HINT}"
            )

        messages = agent.observe(messages, obs)

        if _budget_exhausted(time_used, time_budget, cost_used, cost_budget):
            if type(agent) is ReACTAgent:
                predicted_answer = _force_react_boxed_answer(
                    agent=agent,
                    messages=messages,
                    parse_retries=parse_retries,
                    print_episode=print_episode,
                )
            else:
                preds = agent.act(messages)
                if print_episode:
                    pretty_print_conversation(messages[-2:])
                response_text = preds[0] if preds else ""
                final_ans = _parse_final_answer(response_text)
                if final_ans is not None:
                    predicted_answer = final_ans
            break

    success = _check_answer(predicted_answer, gold_answer)
    return success, step + 1, time_used, cost_used, predicted_answer, gold_answer, messages


def _run_episode_cached(
    example_idx: int,
    ep_idx: int,
    args: argparse.Namespace,
    time_budget: Optional[float],
    cost_budget: Optional[float],
    cache_root: str,
    do_print: bool = False,
) -> Dict[str, Any]:
    """Run a single episode with caching."""
    cache_name = (
        f"ex{example_idx + 1}_ep{ep_idx + 1}"
        f"_{_constraint_tag(time_budget, cost_budget)}.json"
    )
    cache_path = os.path.join(cache_root, cache_name)
    cached = _read_episode_cache(cache_path)
    if cached is not None:
        return cached

    env = GsmHardEnv(
        data_path=args.dataset_path,
        deterministic=args.deterministic,
    )
    agent = _build_agent(args, args.agent)

    success, steps, time_used, cost_used, predicted, gold, messages = _run_episode(
        env=env,
        agent=agent,
        example_index=example_idx,
        time_budget=time_budget,
        cost_budget=cost_budget,
        max_steps=args.max_steps,
        parse_retries=args.parse_retries,
        print_episode=do_print,
        deterministic=args.deterministic,
    )

    result = {
        "example_index": example_idx,
        "episode": ep_idx + 1,
        "time_budget": time_budget,
        "cost_budget": cost_budget,
        "is_success": success,
        "steps": steps,
        "time_used": time_used,
        "cost_used": cost_used,
        "predicted_answer": predicted,
        "gold_answer": gold,
        "messages": messages,
    }
    save_json(cache_path, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run arithmetic tool experiments on MuSiQue dataset."
    )
    ap.add_argument(
        "--out_dir", type=str, default="results",
        help="Output directory for results.",
    )
    ap.add_argument(
        "--dataset_path", type=str,
        default=str(ROOT_DIR / "data" / "gsm_hard_train_processed_sampled.json"),
        help="Path to GSM-Hard dataset JSON.",
    )
    ap.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes per example per constraint.",
    )
    ap.add_argument(
        "--time_constraints", type=float, nargs="+",
        default=DEFAULT_TIME_CONSTRAINTS,
        help="Time budget values.",
    )
    ap.add_argument(
        "--no_time_constraint",
        action="store_true",
        default=False,
        help="Disable time constraints (ignore time budget in feedback and stopping).",
    )
    ap.add_argument(
        "--cost_constraints", type=float, nargs="+",
        default=DEFAULT_COST_CONSTRAINTS,
        help="Cost budget values (must match length of time_constraints).",
    )
    ap.add_argument(
        "--no_cost_constraint",
        action="store_true",
        default=False,
        help="Disable cost constraints (ignore cost budget in feedback and stopping).",
    )
    ap.add_argument(
        "--model", type=str, default="gpt-4.1-mini",
        help="LLM model name.",
    )
    ap.add_argument(
        "--fine_tuned_model", type=str, default=None,
        help="Optional fine-tuned model.",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature.",
    )
    ap.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="Repetition penalty.",
    )
    ap.add_argument(
        "--agent", type=str, default="react",
        choices=["react", "pa", "ours_graph_ilp"],
        help="Agent type to run.",
    )
    ap.add_argument(
        "--max_steps", type=int, default=20,
        help="Max interaction steps per episode.",
    )
    ap.add_argument(
        "--parse_retries", type=int, default=3,
        help="Retries when action parsing fails.",
    )
    ap.add_argument(
        "--num_plans", type=int, default=4,
        help="Number of plans for ours_graph_ilp agent.",
    )
    ap.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic (mean) tool costs/times.",
    )
    ap.add_argument(
        "--stochastic", action="store_true", default=False,
        help="Use stochastic (sampled) tool costs/times.",
    )
    ap.add_argument(
        "--num_jobs", type=int, default=1,
        help="Number of parallel jobs (joblib).",
    )
    ap.add_argument(
        "--joblib_backend", type=str, default="threading",
        choices=["threading", "loky"],
        help="joblib backend.",
    )
    ap.add_argument(
        "--print_episodes", type=int, default=-1,
        help="Pretty-print the first N episodes (-1 for all).",
    )
    ap.add_argument(
        "--max_examples", type=int, default=-1,
        help="Max number of examples to evaluate (-1 for all).",
    )

    args = ap.parse_args()
    if args.stochastic:
        args.deterministic = False
    if args.no_time_constraint:
        args.time_constraints = None
    if args.no_cost_constraint:
        args.cost_constraints = None
    ensure_dir(args.out_dir)

    time_constraints = args.time_constraints
    cost_constraints = args.cost_constraints
    if time_constraints is None and cost_constraints is None:
        constraint_pairs = [(None, None)]
    elif time_constraints is None:
        constraint_pairs = [(None, c) for c in cost_constraints]
    elif cost_constraints is None:
        constraint_pairs = [(t, None) for t in time_constraints]
    else:
        if len(time_constraints) != len(cost_constraints):
            raise ValueError(
                f"time_constraints ({len(time_constraints)}) and "
                f"cost_constraints ({len(cost_constraints)}) must have the same length."
            )
        constraint_pairs = list(zip(time_constraints, cost_constraints))

    agent_dir = args.agent
    if args.agent == "ours_graph_ilp":
        agent_dir = f"{args.agent}_n_{args.num_plans}"
    cache_root = os.path.join(
        args.out_dir, "arithmetic", args.model, agent_dir,
    )
    ensure_dir(cache_root)
    print(f"[arithmetic] cache_root: {cache_root}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    examples = dataset.get("examples", [])
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    n_examples = len(examples)
    print(f"[arithmetic] {n_examples} examples, "
          f"{len(constraint_pairs)} constraint pairs, "
          f"{args.episodes} episodes each")

    if args.num_jobs > 1 and not JOBLIB_AVAILABLE:
        print("[arithmetic] joblib not available; falling back to --num_jobs 1.")
        args.num_jobs = 1
    if args.num_jobs > 1 and args.print_episodes != 0:
        print("[arithmetic] print_episodes disabled when num_jobs > 1.")
        args.print_episodes = 0

    all_rows: List[Dict[str, Any]] = []
    total_succ = 0
    total_eps = 0
    start_time = time.perf_counter()
    printed = 0

    for ci, (t_budget, c_budget) in enumerate(constraint_pairs):
        constraint_tag = _constraint_tag(t_budget, c_budget)
        labels: List[str] = []
        if _time_enabled(t_budget):
            labels.append(f"time={t_budget}")
        if _cost_enabled(c_budget):
            labels.append(f"cost={c_budget}")
        if not labels:
            labels.append("no budget constraints")
        print(
            f"\n[arithmetic] constraint {ci + 1}/{len(constraint_pairs)}: "
            + ", ".join(labels)
        )

        constraint_rows: List[Dict[str, Any]] = []
        constraint_succ = 0
        constraint_eps = 0

        if args.num_jobs > 1:
            assert Parallel is not None and delayed is not None
            tasks = []
            for ex_idx in range(n_examples):
                for ep_idx in range(args.episodes):
                    tasks.append((ex_idx, ep_idx))
            results = Parallel(n_jobs=args.num_jobs, backend=args.joblib_backend)(
                delayed(_run_episode_cached)(
                    ex_idx, ep_idx, args, t_budget, c_budget, cache_root, False,
                )
                for ex_idx, ep_idx in tasks
            )
            for result in results:
                ok = bool(result.get("is_success", False))
                constraint_succ += int(ok)
                constraint_eps += 1
                constraint_rows.append({
                    "example_index": result["example_index"],
                    "episode": result["episode"],
                    "time_budget": t_budget,
                    "cost_budget": c_budget,
                    "is_success": int(ok),
                    "steps": result.get("steps", 0),
                    "time_used": result.get("time_used", 0),
                    "cost_used": result.get("cost_used", 0),
                    "predicted_answer": result.get("predicted_answer"),
                    "gold_answer": result.get("gold_answer"),
                })
        else:
            for ex_idx in range(n_examples):
                for ep_idx in range(args.episodes):
                    item_start = time.perf_counter()
                    do_print = args.print_episodes < 0 or printed < args.print_episodes

                    if do_print:
                        ep_labels: List[str] = []
                        if _time_enabled(t_budget):
                            ep_labels.append(f"time_budget={t_budget}")
                        if _cost_enabled(c_budget):
                            ep_labels.append(f"cost_budget={c_budget}")
                        if not ep_labels:
                            ep_labels.append("no_budget_constraints")
                        print(
                            f"\n[Episode] ex={ex_idx + 1}/{n_examples} "
                            f"ep={ep_idx + 1}/{args.episodes} "
                            + " ".join(ep_labels)
                        )

                    result = _run_episode_cached(
                        ex_idx, ep_idx, args, t_budget, c_budget, cache_root, do_print,
                    )
                    if do_print:
                        printed += 1

                    ok = bool(result.get("is_success", False))
                    constraint_succ += int(ok)
                    constraint_eps += 1

                    constraint_rows.append({
                        "example_index": result["example_index"],
                        "episode": result["episode"],
                        "time_budget": t_budget,
                        "cost_budget": c_budget,
                        "is_success": int(ok),
                        "steps": result.get("steps", 0),
                        "time_used": result.get("time_used", 0),
                        "cost_used": result.get("cost_used", 0),
                        "predicted_answer": result.get("predicted_answer"),
                        "gold_answer": result.get("gold_answer"),
                    })

                    item_elapsed = time.perf_counter() - item_start
                    elapsed = time.perf_counter() - start_time
                    done_count = len(all_rows) + len(constraint_rows)
                    total_count = n_examples * args.episodes * len(constraint_pairs)
                    if done_count > 0:
                        avg = elapsed / done_count
                        eta = avg * (total_count - done_count)
                    else:
                        eta = 0
                    print(
                        f"[arithmetic] ex={ex_idx + 1}/{n_examples} "
                        f"ep={ep_idx + 1}/{args.episodes} "
                        f"ok={ok} "
                        f"item={_format_duration(item_elapsed)} "
                        f"elapsed={_format_duration(elapsed)} "
                        f"ETA={_format_duration(eta)}"
                    )

        total_succ += constraint_succ
        total_eps += constraint_eps
        all_rows.extend(constraint_rows)

        if constraint_eps > 0:
            rate = constraint_succ / constraint_eps
            print(
                f"[arithmetic] constraint {constraint_tag}: "
                f"success_rate={rate:.4f} ({constraint_succ}/{constraint_eps})"
            )

    # Save per-constraint CSVs
    for t_budget, c_budget in constraint_pairs:
        constraint_tag = _constraint_tag(t_budget, c_budget)
        rows_for_constraint = [
            r for r in all_rows
            if r["time_budget"] == t_budget and r["cost_budget"] == c_budget
        ]
        if rows_for_constraint:
            csv_path = os.path.join(cache_root, f"eval_{constraint_tag}.csv")
            save_csv(csv_path, rows_for_constraint)

    # Save combined CSV
    combined_path = os.path.join(cache_root, "eval_all.csv")
    save_csv(combined_path, all_rows)
    print(f"\n[Saved] {combined_path}")

    # Save summary CSV
    summary_rows: List[Dict[str, Any]] = []
    for t_budget, c_budget in constraint_pairs:
        rows_for_constraint = [
            r for r in all_rows
            if r["time_budget"] == t_budget and r["cost_budget"] == c_budget
        ]
        n = len(rows_for_constraint)
        if n > 0:
            succ = sum(r["is_success"] for r in rows_for_constraint)
            avg_steps = sum(r["steps"] for r in rows_for_constraint) / n
            avg_time = sum(r["time_used"] for r in rows_for_constraint) / n
            avg_cost = sum(r["cost_used"] for r in rows_for_constraint) / n
            summary_rows.append({
                "time_budget": t_budget,
                "cost_budget": c_budget,
                "episodes": n,
                "success_rate": f"{succ / n:.4f}",
                "avg_steps": f"{avg_steps:.2f}",
                "avg_time_used": f"{avg_time:.4f}",
                "avg_cost_used": f"{avg_cost:.4f}",
            })
    summary_path = os.path.join(cache_root, "eval_summary.csv")
    save_csv(summary_path, summary_rows)
    print(f"[Saved] {summary_path}")

    if total_eps > 0:
        overall = total_succ / total_eps
        print(f"[Overall] success_rate={overall:.4f} ({total_succ}/{total_eps})")


if __name__ == "__main__":
    main()

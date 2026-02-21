"""
Run ReAct on a prebuilt Sokoban dataset, filtered by chosen T* targets.
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
import random

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.sokoban import SokobanEnv
from real_agents.react_agent import ReACTAgent
from real_agents.pa_agent import PAAgent
from real_agents.ours_graph_ilp import OursGraphILPAgent
from real_agents.ours_graph_ilp_w_o_replanning import OursGraphILPAgentWOReplanning
from real_agents.ours_graph_ilp_w_o_solver import OursGraphILPAgentWOSolver
from real_agents.ours_graph_ilp_w_o_strong_conditioning import OursGraphILPAgentWOStrongConditioning
from real_agents.ours_graph_ilp_w_o_replanning_strong_conditioning import OursGraphILPAgentWOReplanningStrongConditioning
from real_agents.ours_graph_ilp_w_o_solver_replanning import OursGraphILPAgentWOSolverReplanning
from real_agents.ours_graph_ilp_w_o_strong_conditioning_solver import OursGraphILPAgentWOStrongConditioningSolver
from real_agents.ours_graph_ilp_w_o_all import OursGraphILPAgentWOAll
from real_agents.ours_graph_ilp_all_step_replanning import OursGraphILPAgentAllStepReplanning
from utils.llm import pretty_print_conversation
from utils.io import load_dataset, save_csv, save_json, ensure_dir
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore
    JOBLIB_AVAILABLE = False


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


def _render_state(env: SokobanEnv) -> str:
    def to_xy(rc: Tuple[int, int]) -> Tuple[int, int]:
        row, col = rc
        # x is horizontal (col), y is vertical (bottom-left origin): R/L change x, U/D change y.
        x = col
        y = env.H - 1 - row
        return (x, y)

    def fmt_list(xs: List[Tuple[int, int]]) -> str:
        if not xs:
            return "none"
        return ", ".join(f"({x}, {y})" for x, y in xs)

    walls = sorted((to_xy(rc) for rc in env.walls), key=lambda p: (p[1], p[0]))
    goals = sorted((to_xy(rc) for rc in env.goals), key=lambda p: (p[1], p[0]))
    boxes = {to_xy(rc) for rc in env.state.boxes}
    box_on_goal = sorted([p for p in boxes if p in set(goals)], key=lambda p: (p[1], p[0]))
    box_only = sorted([p for p in boxes if p not in set(goals)], key=lambda p: (p[1], p[0]))
    player = to_xy(env.state.player)

    lines = [
        f"wall location: {fmt_list(walls)}",
        f"player location: ({player[0]}, {player[1]})",
        f"box location: {fmt_list(box_only)}",
        f"goal location: {fmt_list(goals)}",
        f"box on goal location: {fmt_list(box_on_goal)}",
    ]
    return "\n".join(lines)


def _assign_bucket(t_star: int, targets: List[int], tol: int) -> Optional[int]:
    if not targets:
        return t_star
    closest = min(targets, key=lambda x: abs(t_star - x))
    if abs(t_star - closest) <= tol:
        return closest
    return None

def _parse_action(text: str) -> Optional[str]:
    match = re.search(r"Action\s*:\s*([A-Za-z_]+)", text, re.IGNORECASE)
    token = match.group(1) if match else None
    if token is None:
        match = re.search(
            r"\b(push_up|push_down|push_left|push_right|move_up|move_down|move_left|move_right|up|down|left|right|u|d|l|r)\b",
            text,
            re.IGNORECASE,
        )
        token = match.group(1) if match else None
    if token is None:
        return None
    token = token.strip().upper()
    mapping = {
        "U": "U",
        "UP": "U",
        "MOVE_UP": "U",
        "PUSH_UP": "U",
        "D": "D",
        "DOWN": "D",
        "MOVE_DOWN": "D",
        "PUSH_DOWN": "D",
        "L": "L",
        "LEFT": "L",
        "MOVE_LEFT": "L",
        "PUSH_LEFT": "L",
        "R": "R",
        "RIGHT": "R",
        "MOVE_RIGHT": "R",
        "PUSH_RIGHT": "R",
    }
    return mapping.get(token)


def _build_react_agent(args) -> ReACTAgent:
    return ReACTAgent(
        model=args.model,
        fine_tuned_model=args.fine_tuned_model,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        instruct_tuned=True,
        debug_mode=False,
        is_print=False,
    )


def _build_pa_agent(args) -> PAAgent:
    return PAAgent(
        model=args.model,
        fine_tuned_model=args.fine_tuned_model,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        instruct_tuned=True,
        debug_mode=False,
        is_print=False,
    )


def _build_agent(args, agent_type: str):
    if agent_type in ("react", "react-best-of-n"):
        return _build_react_agent(args)
    if agent_type in ("pa", "pa-best-of-n"):
        return _build_pa_agent(args)
    if agent_type == "ours_graph_ilp":
        return OursGraphILPAgent(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type == "ours_graph_ilp_w_o_replanning":
        return OursGraphILPAgentWOReplanning(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type == "ours_graph_ilp_w_o_solver":
        return OursGraphILPAgentWOSolver(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type == "ours_graph_ilp_w_o_strong_conditioning":
        return OursGraphILPAgentWOStrongConditioning(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type == "ours_graph_ilp_w_o_replanning_strong_conditioning":
        return OursGraphILPAgentWOReplanningStrongConditioning(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type =="ours_graph_ilp_w_o_solver_replanning":
        return OursGraphILPAgentWOSolverReplanning(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type == "ours_graph_ilp_w_o_strong_conditioning_solver":
        return OursGraphILPAgentWOStrongConditioningSolver(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    if agent_type == "ours_graph_ilp_w_o_all":
        return OursGraphILPAgentWOAll(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
            num_plans=args.num_plans,
        )
    return _build_react_agent(args)


def _read_episode_cache(cache_path: str, t_star: int, t_bucket: int, B: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)
    if cached.get("T_star") != t_star or cached.get("T_bucket") != t_bucket or cached.get("B") != B:
        return None
    return cached


def _run_episode(
    env: SokobanEnv,
    agent: Union[
        ReACTAgent,
        PAAgent,
        OursGraphILPAgent,
        OursGraphILPAgentWOReplanning,
        OursGraphILPAgentWOReplanningStrongConditioning,
        OursGraphILPAgentWOSolver,
        OursGraphILPAgentWOStrongConditioning,
        OursGraphILPAgentWOAll
    ],
    B: int,
    max_tokens: int,
    parse_retries: int,
    act_n: int = 1,
    print_episode: bool = False,
) -> Tuple[bool, int, int, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run a single episode.
    
    Returns:
        Tuple of (success, steps, invalid_count, messages, replanning_info)
        replanning_info is None for agents that don't support it.
    """
    env.reset()
    agent.is_print = print_episode

    plan_text = agent.generate_plan(_render_state(env), B)

    messages = [{"role": "system", "content": agent.generate_initial_message(plan_text=plan_text)}]
    
    if print_episode:
        pretty_print_conversation(messages)
    invalid = 0

    for t in range(B):
        obs = _render_state(env)
        remaining = B - t
        messages = agent.observe(messages, f"{obs}\nSteps remaining: {remaining}")
        if act_n > 1 and isinstance(agent, (ReACTAgent, PAAgent)):
            preds = agent.act(messages, max_tokens=max_tokens, n=act_n)
        else:
            preds = agent.act(messages, max_tokens=max_tokens)
        if print_episode:
            pretty_print_conversation(messages[-2:])
        action = _parse_action(preds[0])
        retry = 0
        while action is None and retry < parse_retries:
            retry += 1
            if messages and messages[-1].get("role") == "assistant":
                messages.pop()
            if act_n > 1 and isinstance(agent, (ReACTAgent, PAAgent)):
                preds = agent.act(messages, max_tokens=max_tokens, n=act_n)
            else:
                preds = agent.act(messages, max_tokens=max_tokens)
            if print_episode:
                pretty_print_conversation(messages[-2:])
            action = _parse_action(preds[0])
        if action is None:
            invalid += 1
        env.step(action)
        if env.is_goal(env.state):
            replanning_info = None
            if hasattr(agent, 'get_replanning_info'):
                replanning_info = agent.get_replanning_info()
            return True, t + 1, invalid, messages, replanning_info
    replanning_info = None
    if hasattr(agent, 'get_replanning_info'):
        replanning_info = agent.get_replanning_info()
    return False, B, invalid, messages, replanning_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory for CSVs.")
    ap.add_argument("--dataset_path", type=str, default="data/dataset.json", help="Path to dataset JSON.")

    ap.add_argument("--episodes", type=int, default=10, help="Episodes per map.")
    ap.add_argument("--slack", type=int, default=2, help="Budget slack added to T* (B = T* + slack).")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name.")
    ap.add_argument("--fine_tuned_model", type=str, default=None, help="Optional fine-tuned model.")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    ap.add_argument(
        "--agent",
        type=str,
        default="ours_graph_ilp",
        choices=[
            "react",
            "react-best-of-n",
            "pa",
            "pa-best-of-n",
            "ours_graph_ilp",
            "ours_graph_ilp_w_o_replanning",
            "ours_graph_ilp_w_o_solver",
            "ours_graph_ilp_w_o_strong_conditioning",
            "ours_graph_ilp_w_o_replanning_strong_conditioning",
            "ours_graph_ilp_w_o_solver_replanning",
            "ours_graph_ilp_w_o_strong_conditioning_solver",
            "ours_graph_ilp_w_o_all",
            "ours_graph_ilp_all_step_replanning"
        ],
        help="Agent type to run.")
    ap.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per LLM step.")
    ap.add_argument("--parse_retries", type=int, default=5,
                    help="Retries when action parsing fails.")
    ap.add_argument("--num_plans", type=int, default=4,
                    help="Number of diverse plans for OursGraphILPAgent or samples for best-of-n agents.")
    ap.add_argument("--num_jobs", type=int, default=10,
                    help="Number of parallel jobs for episodes (joblib).")
    ap.add_argument("--joblib_backend", type=str, default="threading",
                    choices=["threading", "loky"],
                    help="joblib backend; 'loky' uses multiprocessing.")
    ap.add_argument("--print_episodes", type=int, default=-1,
                    help="Pretty-print the first N episodes (-1 for all).")

    ap.add_argument("--T_targets", type=int, nargs="+", default=[6],
                    help="Subset of T* targets to evaluate (default: use dataset meta).")
    ap.add_argument("--T_tol", type=int, default=None, help="Tolerance around T* targets (default: meta).")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    agent_dir = args.agent
    if args.agent.startswith("ours_graph_ilp") or args.agent in ("react-best-of-n", "pa-best-of-n"):
        agent_dir = f"{args.agent}_n_{args.num_plans}"
    if args.T_targets[0] == 6:
        cache_root = os.path.join(args.out_dir, f"sokoban_{args.slack}", args.model, agent_dir)
    else:
        cache_root = os.path.join(args.out_dir, f"sokoban_{args.slack}_T_{args.T_targets}", args.model, agent_dir)
    
    ensure_dir(cache_root)
    print(cache_root)

    ds = load_dataset(args.dataset_path)
    data = ds["data"]
    meta: Dict[str, Any] = ds.get("meta", {})

    chosen_targets = args.T_targets
    if chosen_targets is None:
        chosen_targets = [int(t) for t in meta.get("T_targets", [])]
    chosen_targets = [int(t) for t in chosen_targets]

    tol = args.T_tol if args.T_tol is not None else int(meta.get("T_tol", 0))

    chosen: List[Dict[str, Any]] = []
    for item in data:
        t_star = int(item["T_star"])
        bucket = _assign_bucket(t_star, chosen_targets, tol)
        if bucket is None:
            continue
        item = dict(item)
        item["T_bucket"] = bucket
        chosen.append(item)

    if not chosen:
        raise RuntimeError("No maps matched the requested T_targets (check T_targets/T_tol).")

    if args.num_jobs > 1 and not JOBLIB_AVAILABLE:
        print("[ReAct] joblib not available; falling back to --num_jobs 1.")
        args.num_jobs = 1

    if args.num_jobs > 1 and args.print_episodes != 0:
        print("[ReAct] print_episodes is disabled when num_jobs > 1.")
        args.print_episodes = 0

    dataset_stem = os.path.splitext(os.path.basename(args.dataset_path))[0]
    targets_tag = "-".join(str(t) for t in sorted(set(chosen_targets))) if chosen_targets else "all"

    rows: List[Dict[str, Any]] = []
    total_succ = 0
    total_eps = 0
    total = len(chosen)
    start_time = time.perf_counter()
    printed = 0
    for idx, item in enumerate(chosen, start=1):
        item_start = time.perf_counter()
        mp = item["map"]
        t_star = int(item["T_star"])
        t_bucket = int(item["T_bucket"])
        B = t_star + args.slack

        def run_episode_cached(ep_idx: int, do_print: bool) -> Tuple[bool, int]:
            cache_name = f"{dataset_stem}_T{targets_tag}_idx{idx}_ep{ep_idx + 1}.json"
            cache_path = os.path.join(cache_root, cache_name)
            cached = _read_episode_cache(cache_path, t_star, t_bucket, B)
            if cached is not None:
                if do_print and cached.get("messages"):
                    pretty_print_conversation(cached["messages"])
                return bool(cached.get("is_success", False)), int(cached.get("step", B))

            env = SokobanEnv(mp)
            # Create a fresh agent for each episode to avoid state sharing issues
            local_agent = _build_agent(args, args.agent)
            if isinstance(
                local_agent,
                (
                    OursGraphILPAgent,
                    OursGraphILPAgentWOReplanning,
                    OursGraphILPAgentWOReplanningStrongConditioning,
                    OursGraphILPAgentWOSolver,
                    OursGraphILPAgentWOStrongConditioning,
                    OursGraphILPAgentWOSolverReplanning,
                    OursGraphILPAgentWOStrongConditioningSolver,
                    OursGraphILPAgentWOAll,
                    OursGraphILPAgentAllStepReplanning,
                ),
            ):
                graph_base = os.path.splitext(cache_path)[0]
                local_agent.set_graph_log_base(graph_base)
            act_n = max(1, args.num_plans) if args.agent in ("react-best-of-n", "pa-best-of-n") else 1
            ok, steps, _invalid, messages, replanning_info = _run_episode(
                env=env,
                agent=local_agent,
                B=B,
                max_tokens=args.max_tokens,
                parse_retries=args.parse_retries,
                act_n=act_n,
                print_episode=do_print,
            )
            cache_data = {
                "T_star": t_star,
                "T_bucket": t_bucket,
                "B": B,
                "is_success": ok,
                "step": steps,
                "messages": messages,
            }
            # Add replanning info if available
            if replanning_info is not None:
                cache_data["num_replanning"] = replanning_info.get("num_replanning", 0)
                cache_data["replanning_states"] = replanning_info.get("replanning_states", [])
            save_json(cache_path, cache_data)
            return bool(ok), int(steps)

        succ = 0
        steps_list: List[int] = []
        if args.num_jobs > 1:
            assert Parallel is not None and delayed is not None
            results = Parallel(n_jobs=args.num_jobs, backend=args.joblib_backend)(
                delayed(run_episode_cached)(ep_idx, False) for ep_idx in range(args.episodes)
            )
            for ok, steps in results:
                succ += int(ok)
                steps_list.append(steps)
        else:
            for ep_idx in range(args.episodes):
                do_print = args.print_episodes < 0 or printed < args.print_episodes
                if do_print:
                    print(f"\n[Episode] map={idx}/{total} ep={ep_idx + 1}/{args.episodes} T*={t_star} B={B}")
                ok, steps = run_episode_cached(ep_idx, do_print)
                if do_print:
                    printed += 1
                succ += int(ok)
                steps_list.append(steps)

        row = {
            "T_star": t_star,
            "T_bucket": t_bucket,
            "B": B,
            "episodes": args.episodes,
            "react_succ": f"{(succ / args.episodes):.4f}",
            "react_avg_steps": sum(steps_list) / len(steps_list),
        }
        rows.append(row)
        total_succ += succ
        total_eps += args.episodes

        item_elapsed = time.perf_counter() - item_start
        elapsed = time.perf_counter() - start_time
        avg = elapsed / idx
        eta = avg * (total - idx)
        print(
            f"[ReAct] {idx}/{total} T*={t_star} bucket={t_bucket} B={B} "
            f"item={_format_duration(item_elapsed)} "
            f"elapsed={_format_duration(elapsed)} ETA={_format_duration(eta)}"
        )

    rows.sort(key=lambda x: (x["T_bucket"], x["T_star"]))
    save_path = os.path.join(cache_root, f"react_eval_T{targets_tag}.csv")
    save_csv(save_path, rows)
    print(f"[Saved] {save_path}")
    if total_eps > 0:
        overall = total_succ / total_eps
        print(f"[Overall] success_rate={overall:.4f} ({total_succ}/{total_eps})")


if __name__ == "__main__":
    main()

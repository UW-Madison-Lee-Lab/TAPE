"""
Run ReAct on the ALFWorld environment.
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

from env.alfworld import AlfworldEnv, get_gold_plan_length
from real_agents.react_agent import ReACTAgent
from real_agents.pa_agent import PAAgent
from real_agents.ours_graph_ilp import OursGraphILPAgentAlfworld
from utils.llm import pretty_print_conversation
from utils.io import save_csv, save_json, ensure_dir
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


def _read_episode_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_gamefiles(info: Any) -> List[str]:
    if not isinstance(info, dict):
        return []
    gamefiles = info.get("extra.gamefile")
    if isinstance(gamefiles, (list, tuple)):
        return [gf for gf in gamefiles if gf]
    if isinstance(gamefiles, str):
        return [gamefiles]
    return []


def _min_gold_plan_length(gamefiles: List[str]) -> int:
    if not gamefiles:
        return -1
    lengths = [get_gold_plan_length(gf) for gf in gamefiles]
    return min(lengths)


def _parse_initial_obs(obs: str) -> Tuple[str, str]:
    header = "-= Welcome to TextWorld, ALFRED! =-\n\n"
    if obs.startswith(header):
        obs = obs[len(header):]
    obs = obs.strip()
    if "\n\nYour task is to: " in obs:
        observation, task = obs.split("\n\nYour task is to: ", 1)
    else:
        observation, task = obs, ""
    observation = observation.replace("Initial Observation: ", "").strip()
    task = task.strip()
    return observation, task


def _clean_action(action: str) -> Optional[str]:
    if action is None:
        return None
    action = action.strip().strip("\"").strip("'")
    action = action.rstrip(".").strip()
    return action or None


def _parse_action(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"Action\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return _clean_action(match.group(1).splitlines()[0])

    patterns = [
        r"\bgo to [^\n]+",
        r"\btake [^\n]+ from [^\n]+",
        r"\bput [^\n]+ (?:in|on) [^\n]+",
        r"\bopen [^\n]+",
        r"\bclose [^\n]+",
        r"\btoggle [^\n]+ [^\n]+",
        r"\bclean [^\n]+ with [^\n]+",
        r"\bheat [^\n]+ with [^\n]+",
        r"\bcool [^\n]+ with [^\n]+",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return _clean_action(m.group(0))
    return None


def _format_initial_observation(
    observation: str,
    task: str,
    remaining: int,
) -> str:
    parts = []
    if task:
        parts.append(f"Your task is to: {task}")
    parts.append(observation)
    parts.append(f"Steps remaining: {remaining}")
    return "\n".join(parts)


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
            task="alfworld",
        )
    if agent_type == "ours_graph_ilp":
        return OursGraphILPAgentAlfworld(
            model=args.model,
            fine_tuned_model=args.fine_tuned_model,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            instruct_tuned=True,
            debug_mode=False,
            is_print=False,
        )
    return ReACTAgent(
        model=args.model,
        fine_tuned_model=args.fine_tuned_model,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        instruct_tuned=True,
        debug_mode=False,
        is_print=False,
        task="alfworld",
    )


def _env_step(env: AlfworldEnv, action: str):
    try:
        return env.step([action])
    except Exception:
        return env.step(action)


def _unpack_step(step_out: Any) -> Tuple[str, bool, Any]:
    obs_list: Any = None
    info: Any = {}
    dones: Any = False
    if isinstance(step_out, tuple) and len(step_out) >= 4:
        obs_list = step_out[0]
        info = step_out[3]
        dones = step_out[2]
    else:
        obs_list = step_out
    obs = ""
    if isinstance(obs_list, (list, tuple)) and obs_list:
        obs = obs_list[0]
    elif isinstance(obs_list, str):
        obs = obs_list
    done = False
    if isinstance(dones, (list, tuple)):
        done = bool(dones[0]) if dones else False
    else:
        done = bool(dones)
    return obs, done, info


def _advance_env_to_episode(
    env: AlfworldEnv, episode_index: int
) -> Tuple[List[str], Any, str]:
    obs_list: List[str] = []
    info: Any = {}
    transition_info_text = ""
    for _ in range(episode_index + 1):
        obs_list, info, _recep_to_objs, transition_info_text = env.reset()
    return obs_list, info, transition_info_text


def _run_episode(
    env: AlfworldEnv,
    agent: Union[
        ReACTAgent,
        PAAgent,
        OursGraphILPAgentAlfworld,
    ],
    slack: int,
    max_tokens: int,
    parse_retries: int,
    print_episode: bool = False,
    reset_data: Optional[Tuple[List[str], Any, str]] = None,
) -> Tuple[bool, int, int, int, int, str, Optional[str], List[Dict[str, Any]]]:
    if reset_data is None:
        obs_list, info, _recep_to_objs, transition_info_text = env.reset()
    else:
        obs_list, info, transition_info_text = reset_data
    obs = obs_list[0] if obs_list else ""
    observation, task = _parse_initial_obs(obs)

    gamefiles = _extract_gamefiles(info)
    t_star = _min_gold_plan_length(gamefiles)
    if t_star < 0:
        t_star = 0
    B = t_star + slack

    agent.is_print = print_episode

    plan_obs = observation
    if isinstance(agent, (PAAgent, OursGraphILPAgentAlfworld)) and task:
        plan_obs = f"Your task is to: {task}\n{observation}".strip()
    plan_text = agent.generate_plan(plan_obs, B, transition_info_text=transition_info_text)

    messages = [
        {
            "role": "system",
            "content": agent.generate_initial_message(
                plan_text=plan_text,
                transition_info_text=transition_info_text
            ),
        }
    ]
    if print_episode:
        pretty_print_conversation(messages)

    invalid = 0
    cur_obs = obs

    for t in range(B):
        remaining = B - t
        if t == 0:
            obs_text = _format_initial_observation(
                observation=observation,
                task=task,
                remaining=remaining,
            )
        else:
            obs_text = f"{cur_obs}\nSteps remaining: {remaining}"
        messages = agent.observe(messages, obs_text)
        preds = agent.act(messages, max_tokens=max_tokens)
        if print_episode:
            pretty_print_conversation(messages[-2:])
        action = _parse_action(preds[0])
        retry = 0
        while action is None and retry < parse_retries:
            retry += 1
            if messages and messages[-1].get("role") == "assistant":
                messages.pop()
            preds = agent.act(messages, max_tokens=max_tokens)
            if print_episode:
                pretty_print_conversation(messages[-2:])
            action = _parse_action(preds[0])
        if action is None:
            invalid += 1
            action = "look"

        step_out = _env_step(env, action)
        cur_obs, done, step_info = _unpack_step(step_out)
        if print_episode:
            commands = None
            if isinstance(step_info, dict):
                commands = step_info.get("admissible_commands")
            if isinstance(commands, (list, tuple)):
                print(f"[admissible_commands] {commands}")
        if done:
            return True, t + 1, invalid, B, t_star, task, gamefiles[0] if gamefiles else None, messages

    return False, B, invalid, B, t_star, task, gamefiles[0] if gamefiles else None, messages


def _run_episode_at_index(
    ep_idx: int,
    args: argparse.Namespace,
) -> Tuple[int, bool, int, int, int, int, str, Optional[str], List[Dict[str, Any]]]:
    env = AlfworldEnv(config=args.config, split=args.split, batch_size=1)
    reset_data = _advance_env_to_episode(env, ep_idx)
    agent = _build_agent(args, args.agent)
    ok, steps, invalid, B, t_star, task, gamefile_path, messages = _run_episode(
        env=env,
        agent=agent,
        slack=args.slack,
        max_tokens=args.max_tokens,
        parse_retries=args.parse_retries,
        print_episode=False,
        reset_data=reset_data,
    )
    return ep_idx, ok, steps, invalid, B, t_star, task, gamefile_path, messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory for CSVs.")
    ap.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "data" / "mini_config.yaml"),
        help="ALFWorld config YAML.",
    )
    ap.add_argument("--split", type=str, default="eval_out_of_distribution", help="ALFWorld split.")

    ap.add_argument("--episodes", type=int, default=134, help="Number of episodes.")
    ap.add_argument("--slack", type=int, default=2, help="Budget slack added to T* (B = T* + slack).")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name.")
    ap.add_argument("--fine_tuned_model", type=str, default=None, help="Optional fine-tuned model.")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    ap.add_argument(
        "--agent",
        type=str,
        default="ours_graph_ilp",
        choices=["react", "pa", "ours_graph_ilp"],
        help="Agent type to run.",
    )
    ap.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per LLM step.")
    ap.add_argument("--parse_retries", type=int, default=5, help="Retries when action parsing fails.")
    ap.add_argument("--print_episodes", type=int, default=-1, help="Pretty-print the first N episodes (-1 for all).")
    ap.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs for episodes (joblib).")
    ap.add_argument(
        "--joblib_backend",
        type=str,
        default="threading",
        choices=["threading", "loky"],
        help="joblib backend; 'loky' uses multiprocessing.",
    )

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    cache_root = os.path.join(args.out_dir, f"alfworld_{args.split}_{args.slack}", args.model, args.agent)
    ensure_dir(cache_root)
    print(cache_root)

    env = AlfworldEnv(config=args.config, split=args.split, batch_size=1)
    agent = _build_agent(args, args.agent)

    rows: List[Dict[str, Any]] = []
    total_succ = 0
    total_eps = 0
    start_time = time.perf_counter()
    printed = 0

    if args.num_jobs > 1 and not JOBLIB_AVAILABLE:
        print("[ReAct] joblib not available; falling back to --num_jobs 1.")
        args.num_jobs = 1

    if args.num_jobs > 1 and args.print_episodes != 0:
        print("[ReAct] print_episodes is disabled when num_jobs > 1.")
        args.print_episodes = 0

    if args.num_jobs > 1:
        assert Parallel is not None and delayed is not None
        results = Parallel(n_jobs=args.num_jobs, backend=args.joblib_backend)(
            delayed(_run_episode_at_index)(ep_idx, args) for ep_idx in range(args.episodes)
        )
        for ep_idx, ok, steps, invalid, B, t_star, task, gamefile_path, messages in sorted(
            results, key=lambda x: x[0]
        ):
            cache_name = f"ep{ep_idx + 1}.json"
            cache_path = os.path.join(cache_root, cache_name)
            save_json(
                cache_path,
                {
                    "episode": ep_idx + 1,
                    "T_star": t_star,
                    "B": B,
                    "is_success": ok,
                    "step": steps,
                    "invalid": invalid,
                    "task": task,
                    "gamefile": gamefile_path,
                    "messages": messages,
                },
            )

            rows.append(
                {
                    "episode": ep_idx + 1,
                    "T_star": t_star,
                    "B": B,
                    "is_success": int(ok),
                    "step": steps,
                    "invalid": invalid,
                }
            )
            total_succ += int(ok)
            total_eps += 1
        elapsed = time.perf_counter() - start_time
        print(f"[ReAct] completed {args.episodes} episodes in {_format_duration(elapsed)}")
    else:
        for ep_idx in range(args.episodes):
            item_start = time.perf_counter()
            do_print = args.print_episodes < 0 or printed < args.print_episodes
            if do_print:
                print(f"\n[Episode] ep={ep_idx + 1}/{args.episodes}")

            cache_name = f"ep{ep_idx + 1}.json"
            cache_path = os.path.join(cache_root, cache_name)
            cached = _read_episode_cache(cache_path)
            if cached is not None:
                env.reset()
                if do_print:
                    printed += 1
                ok = bool(cached.get("is_success", False))
                steps = int(cached.get("step", cached.get("B", 0) or 0))
                invalid = int(cached.get("invalid", 0))
                t_star = int(cached.get("T_star", 0))
                B = int(cached.get("B", 0))
                task = cached.get("task", "")
                gamefile_path = cached.get("gamefile")
                messages = cached.get("messages", [])
            else:
                ok, steps, invalid, B, t_star, task, gamefile_path, messages = _run_episode(
                    env=env,
                    agent=agent,
                    slack=args.slack,
                    max_tokens=args.max_tokens,
                    parse_retries=args.parse_retries,
                    print_episode=do_print,
                )
                if do_print:
                    printed += 1
                save_json(
                    cache_path,
                    {
                        "episode": ep_idx + 1,
                        "T_star": t_star,
                        "B": B,
                        "is_success": ok,
                        "step": steps,
                        "invalid": invalid,
                        "task": task,
                        "gamefile": gamefile_path,
                        "messages": messages,
                    },
                )

            rows.append(
                {
                    "episode": ep_idx + 1,
                    "T_star": t_star,
                    "B": B,
                    "is_success": int(ok),
                    "step": steps,
                    "invalid": invalid,
                }
            )
            total_succ += int(ok)
            total_eps += 1

            item_elapsed = time.perf_counter() - item_start
            elapsed = time.perf_counter() - start_time
            avg = elapsed / (ep_idx + 1)
            eta = avg * (args.episodes - ep_idx - 1)
            print(
                f"[ReAct] ep={ep_idx + 1}/{args.episodes} T*={t_star} B={B} "
                f"item={_format_duration(item_elapsed)} "
                f"elapsed={_format_duration(elapsed)} ETA={_format_duration(eta)}"
            )

    save_path = os.path.join(cache_root, "alfworld_eval.csv")
    save_csv(save_path, rows)
    print(f"[Saved] {save_path}")
    if total_eps > 0:
        overall = total_succ / total_eps
        print(f"[Overall] success_rate={overall:.4f} ({total_succ}/{total_eps})")


if __name__ == "__main__":
    main()

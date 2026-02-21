from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Any
import os
import time

from env.sokoban import SokobanEnv, BFSPlanner
from agents.react import ReActPolicy
from agents.plan_and_act import PlanAndActPolicy
from agents.ours_graph_ilp import OursGraphILPPolicy
from utils.eval import eval_policy
from utils.io import load_dataset, save_csv
from utils.planning import ORTOOLS_AVAILABLE


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


def run_T_eval(args):
    ds = load_dataset(args.dataset_path)
    data = ds["data"]

    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        T = int(item["T_star"])
        buckets[T].append(item)

    rows: List[Dict[str, Any]] = []
    total = len(data)
    start_time = time.perf_counter()
    for idx, item in enumerate(data, start=1):
        item_start = time.perf_counter()
        mp = item["map"]
        T_star = int(item["T_star"])
        env = SokobanEnv(mp)
        planner = BFSPlanner(env, max_nodes=args.max_nodes)

        B = T_star + args.slack

        react = ReActPolicy(env, planner, delta=args.delta_react, eta=args.eta, seed=1)
        pa = PlanAndActPolicy(
            env,
            planner,
            delta_plan=args.delta_plan,
            delta_react=args.delta_react,
            eta=args.eta,
            p_follow=args.p_follow,
            seed=2,
        )

        r = eval_policy(env, planner, react, episodes=args.episodes, B=B, base_seed=args.seed)
        p = eval_policy(env, planner, pa, episodes=args.episodes, B=B, base_seed=args.seed + 10)

        row = {
            "T_star": T_star,
            "B": B,
            "react_succ": r["success_rate"],
            "pa_succ": p["success_rate"],
            "pa_condition_holds": p.get("pa_condition_holds", None),
            "pa_kappa_hat(fallback)": p.get("error_stats_fallback_only", {}).get("kappa_hat", float("nan")),
            "pa_rho_hat(fallback)": p.get("error_stats_fallback_only", {}).get("rho_hat", float("nan")),
        }

        if args.run_ours:
            if not ORTOOLS_AVAILABLE:
                row["ours_succ"] = None
            else: 
                ours = OursGraphILPPolicy(env, planner, delta=args.delta_react, N=args.N, seed=3)
                o = eval_policy(env, planner, ours, episodes=args.episodes, B=B, base_seed=args.seed + 20)
                row["ours_succ"] = o["success_rate"]
                row["ours_mean_d"] = o.get("mean_d_ut", float("nan"))
                row["ours_mean_V"] = o.get("mean_V_size", float("nan"))
                row["ours_mean_E"] = o.get("mean_E_size", float("nan"))

        rows.append(row)
        item_elapsed = time.perf_counter() - item_start
        elapsed = time.perf_counter() - start_time
        avg = elapsed / idx
        eta = avg * (total - idx)
        print(
            f"[T_eval] {idx}/{total} T*={T_star} B={B} "
            f"item={_format_duration(item_elapsed)} "
            f"elapsed={_format_duration(elapsed)} "
            f"ETA={_format_duration(eta)}"
        )

    rows.sort(key=lambda x: x["T_star"])
    save_path = os.path.join(args.out_dir, "T_eval.csv")
    save_csv(save_path, rows)
    print(f"[Saved] {save_path}")


def run_N_eval(args):
    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not available. Install: pip install ortools")
        return

    ds = load_dataset(args.dataset_path)
    data = ds["data"]

    chosen = [it for it in data if abs(int(it["T_star"]) - args.N_T_target) <= args.N_T_tol]
    chosen = chosen[:args.N_num_maps]
    if not chosen:
        raise RuntimeError("No maps found near N_T_target; rebuild dataset or adjust tolerances.")

    rows: List[Dict[str, Any]] = []
    total_runs = len(args.N_list) * len(chosen)
    run_idx = 0
    start_time = time.perf_counter()
    for N in args.N_list:
        n_start = time.perf_counter()
        succs: List[float] = []
        mean_ds: List[float] = []
        mean_Vs: List[float] = []
        mean_Es: List[float] = []

        for j, item in enumerate(chosen):
            item_start = time.perf_counter()
            mp = item["map"]
            T_star = int(item["T_star"])
            env = SokobanEnv(mp)
            planner = BFSPlanner(env, max_nodes=args.max_nodes)
            B = T_star + args.slack

            ours = OursGraphILPPolicy(env, planner, delta=args.delta_react, N=N, seed=1000 + N)
            o = eval_policy(env, planner, ours, episodes=args.episodes, B=B, base_seed=args.seed + 100 + j * 1000)

            succs.append(o["success_rate"])
            mean_ds.append(o.get("mean_d_ut", float("nan")))
            mean_Vs.append(o.get("mean_V_size", float("nan")))
            mean_Es.append(o.get("mean_E_size", float("nan")))

            run_idx += 1
            item_elapsed = time.perf_counter() - item_start
            elapsed = time.perf_counter() - start_time
            avg = elapsed / run_idx if run_idx else 0.0
            eta = avg * (total_runs - run_idx)
            print(
                f"[N_eval] N={N} map={j + 1}/{len(chosen)} T*={T_star} "
                f"{run_idx}/{total_runs} item={_format_duration(item_elapsed)} "
                f"elapsed={_format_duration(elapsed)} ETA={_format_duration(eta)}"
            )

        rows.append({
            "N": N,
            "num_maps": len(chosen),
            "episodes_per_map": args.episodes,
            "T_target": args.N_T_target,
            "succ_mean": sum(succs) / len(succs),
            "mean_d_ut": sum(mean_ds) / len(mean_ds),
            "mean_V_size": sum(mean_Vs) / len(mean_Vs),
            "mean_E_size": sum(mean_Es) / len(mean_Es),
        })
        n_elapsed = time.perf_counter() - n_start
        print(f"[N_eval] Finished N={N} in {_format_duration(n_elapsed)}")

    save_path = os.path.join(args.out_dir, f"N_eval_T{args.N_T_target}.csv")
    save_csv(save_path, rows)
    print(f"[Saved] {save_path}")

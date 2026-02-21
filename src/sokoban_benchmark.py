"""
Realistic-ish Sokoban benchmark (corridor removed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.dataset import build_dataset_by_Tstar
from utils.experiments import run_T_eval, run_N_eval
from utils.io import ensure_dir
from utils.planning import ORTOOLS_AVAILABLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory for evaluation CSV files.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for dataset generation and eval.")
    ap.add_argument("--max_nodes", type=int, default=200000, help="Max BFS nodes per planning call (oracle).")

    ap.add_argument("--delta_react", type=float, default=0.16, help="ReAct recommendation error rate.")
    ap.add_argument("--eta", type=float, default=0.04, help="Execution noise rate for deviating from intended action.")
    ap.add_argument("--delta_plan", type=float, default=0.16, help="Plan-and-Act planning error rate.")
    ap.add_argument("--p_follow", type=float, default=0.85, help="Plan-and-Act follow-plan probability when aligned.")

    ap.add_argument("--slack", type=int, default=2, help="Budget slack added to T* (B = T* + slack).")

    ap.add_argument("--run_ours", action="store_true", help="Evaluate OursGraphILP policy (requires OR-Tools).")
    ap.add_argument("--N", type=int, default=16, help="Number of subplans for OursGraphILP folding.")

    ap.add_argument("--dataset_path", type=str, default="data/dataset.json", help="Path to dataset JSON (input for eval, output for build).")
    ap.add_argument("--build_dataset", action="store_true", help="Build dataset before evaluation.")
    ap.add_argument("--num_maps_per_bucket", type=int, default=10, help="Maps per T* bucket.")
    ap.add_argument("--T_targets", type=int, nargs="+", default=[6, 10, 14, 18, 22], help="Target T* values for buckets.")
    ap.add_argument("--T_tol", type=int, default=1, help="Tolerance around T* targets.")

    ap.add_argument("--H", type=int, default=7, help="Map height for dataset generation.")
    ap.add_argument("--W", type=int, default=9, help="Map width for dataset generation.")
    ap.add_argument("--num_boxes", type=int, default=1, help="Number of boxes/goals per map.")
    ap.add_argument("--reverse_steps_min", type=int, default=10, help="Minimum reverse-play steps for solvable map generation.")
    ap.add_argument("--reverse_steps_max", type=int, default=60, help="Maximum reverse-play steps for solvable map generation.")
    ap.add_argument("--internal_wall_prob", type=float, default=0.08, help="Probability of internal walls during map generation.")

    ap.add_argument("--episodes", type=int, default=200, help="Episodes per map for evaluation.")
    ap.add_argument("--run_T_eval", action="store_true", help="Run T* sweep evaluation.")
    ap.add_argument("--run_N_eval", action="store_true", help="Run N sweep for OursGraphILP.")

    ap.add_argument("--N_list", type=int, nargs="+", default=[2, 4, 8, 16, 32], help="N values for N-eval sweep.")
    ap.add_argument("--N_T_target", type=int, default=14, help="Target T* for N-eval map filter.")
    ap.add_argument("--N_T_tol", type=int, default=1, help="Tolerance around N_T_target.")
    ap.add_argument("--N_num_maps", type=int, default=5, help="Number of maps for N-eval.")

    args = ap.parse_args()

    if args.build_dataset:
        dataset_parent = Path(args.dataset_path).parent
        ensure_dir(str(dataset_parent))
        build_dataset_by_Tstar(
            out_path=args.dataset_path,
            num_maps_per_bucket=args.num_maps_per_bucket,
            T_targets=args.T_targets,
            T_tol=args.T_tol,
            H=args.H,
            W=args.W,
            num_boxes=args.num_boxes,
            reverse_steps_min=args.reverse_steps_min,
            reverse_steps_max=args.reverse_steps_max,
            internal_wall_prob=args.internal_wall_prob,
            max_nodes=args.max_nodes,
            seed=args.seed,
        )

    if not (args.run_T_eval or args.run_N_eval):
        # Keep backward compatibility for evaluation-only runs, while making
        # `--build_dataset` behave as "build only" unless eval flags are explicit.
        if args.build_dataset:
            return
        args.run_T_eval = True

    ensure_dir(args.out_dir)

    if args.run_T_eval:
        run_T_eval(args)
    if args.run_N_eval:
        run_N_eval(args)

    if args.run_ours and not ORTOOLS_AVAILABLE:
        print("\n[Note] OR-Tools not detected. Install with: pip install ortools")


if __name__ == "__main__":
    main()

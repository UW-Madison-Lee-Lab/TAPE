from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math

from env.sokoban import State, SokobanEnv, BFSPlanner
from agents.react import ReActPolicy
from agents.plan_and_act import PlanAndActPolicy
from agents.ours_graph_ilp import OursGraphILPPolicy


@dataclass
class StepLog:
    intended: str
    executed: str
    intended_viable: bool
    executed_viable: bool
    deviated: bool
    forced: bool
    d_cur: int
    V: int
    E: int
    used_ilp: bool


def estimate_delta_eta_kappa_rho(logs: List[StepLog]) -> Dict[str, Any]:
    n = len(logs)
    if n == 0:
        return {"steps": 0}

    delta_hat = sum(1 for z in logs if not z.intended_viable) / n
    eta_hat = sum(1 for z in logs if z.deviated) / n

    A = [z for z in logs if z.intended_viable and z.deviated]
    B = [z for z in logs if (not z.intended_viable) and z.deviated]

    kappa_hat = (sum(1 for z in A if not z.executed_viable) / len(A)) if len(A) else float("nan")
    rho_hat = (sum(1 for z in B if z.executed_viable) / len(B)) if len(B) else float("nan")

    return {
        "steps": n,
        "delta_hat": delta_hat,
        "eta_hat": eta_hat,
        "kappa_hat": kappa_hat,
        "rho_hat": rho_hat,
        "cnt_A(intended_viable & deviated)": len(A),
        "cnt_B(intended_nonviable & deviated)": len(B),
    }


def mean_or_nan(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def run_episode_collect(
    env: SokobanEnv,
    planner: BFSPlanner,
    policy: Any,
    B: int,
    episode_seed: int,
) -> Tuple[bool, int, List[StepLog], Dict[str, Any]]:
    st0 = env.reset()
    st = st0

    if hasattr(policy, "reset_episode"):
        policy.reset_episode(seed=episode_seed)

    meta: Dict[str, Any] = {}
    logs: List[StepLog] = []

    for t in range(B):
        rem = B - t
        viable = set(planner.viable_actions(st, rem))

        forced = False
        d_cur = -1
        Vn = 0
        En = 0
        used_ilp = False

        if isinstance(policy, ReActPolicy):
            executed, intended = policy.act(st, rem)

        elif isinstance(policy, PlanAndActPolicy):
            executed, intended, forced = policy.act(st, t, st0, rem, B_total=B)
            meta = policy.get_plan_stats()

        elif isinstance(policy, OursGraphILPPolicy):
            executed, info = policy.act(st, rem)
            intended = executed
            d_cur = int(info.get("d_cur", -1))
            Vn = int(info.get("V", 0))
            En = int(info.get("E", 0))
            used_ilp = bool(info.get("used_ilp", False))

        else:
            raise ValueError("Unknown policy type.")

        deviated = (executed != intended)
        intended_viable = (intended in viable)
        executed_viable = (executed in viable)

        logs.append(StepLog(
            intended=intended,
            executed=executed,
            intended_viable=intended_viable,
            executed_viable=executed_viable,
            deviated=deviated,
            forced=forced,
            d_cur=d_cur,
            V=Vn,
            E=En,
            used_ilp=used_ilp,
        ))

        st, _status, _ = env.step(executed)
        if env.is_goal(st):
            return True, t + 1, logs, meta

    return False, B, logs, meta


def eval_policy(
    env: SokobanEnv,
    planner: BFSPlanner,
    policy: Any,
    episodes: int,
    B: int,
    base_seed: int,
) -> Dict[str, Any]:
    succ = 0
    steps_list: List[int] = []

    all_logs: List[StepLog] = []
    all_logs_fallback_only: List[StepLog] = []
    plan_delta_hats: List[float] = []

    ours_d: List[float] = []
    ours_V: List[float] = []
    ours_E: List[float] = []
    ours_used_ilp: List[int] = []

    for i in range(episodes):
        ok, steps, logs, meta = run_episode_collect(env, planner, policy, B=B, episode_seed=base_seed + i)
        succ += int(ok)
        steps_list.append(steps)
        all_logs.extend(logs)

        if isinstance(policy, PlanAndActPolicy):
            fb = [z for z in logs if not z.forced]
            all_logs_fallback_only.extend(fb)
            if "delta_plan_hat" in meta and isinstance(meta["delta_plan_hat"], float):
                if not math.isnan(meta["delta_plan_hat"]):
                    plan_delta_hats.append(meta["delta_plan_hat"])

        if isinstance(policy, OursGraphILPPolicy):
            for z in logs:
                if z.d_cur >= 0:
                    ours_d.append(float(z.d_cur))
                if z.V > 0:
                    ours_V.append(float(z.V))
                if z.E > 0:
                    ours_E.append(float(z.E))
                ours_used_ilp.append(1 if z.used_ilp else 0)

    out: Dict[str, Any] = {}
    out["episodes"] = episodes
    out["B"] = B
    out["success_rate"] = succ / episodes
    out["avg_steps"] = sum(steps_list) / len(steps_list)
    out["error_stats_all"] = estimate_delta_eta_kappa_rho(all_logs)

    if isinstance(policy, PlanAndActPolicy):
        out["error_stats_fallback_only"] = estimate_delta_eta_kappa_rho(all_logs_fallback_only)
        out["delta_plan_hat_mean"] = mean_or_nan(plan_delta_hats)

        es = out["error_stats_fallback_only"]
        delta_hat = es.get("delta_hat", float("nan"))
        kappa_hat = es.get("kappa_hat", float("nan"))
        rho_hat = es.get("rho_hat", float("nan"))
        if not (math.isnan(delta_hat) or math.isnan(kappa_hat) or math.isnan(rho_hat)):
            out["pa_condition_LHS"] = (1.0 - delta_hat) * kappa_hat
            out["pa_condition_RHS"] = delta_hat * rho_hat
            out["pa_condition_holds"] = out["pa_condition_LHS"] >= out["pa_condition_RHS"]
        else:
            out["pa_condition_holds"] = None

    if isinstance(policy, OursGraphILPPolicy):
        out["mean_d_ut"] = mean_or_nan(ours_d)
        out["mean_V_size"] = mean_or_nan(ours_V)
        out["mean_E_size"] = mean_or_nan(ours_E)
        out["ilp_used_rate"] = (sum(ours_used_ilp) / len(ours_used_ilp)) if ours_used_ilp else float("nan")

    return out

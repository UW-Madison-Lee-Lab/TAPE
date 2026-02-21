from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable, Union
import random
import math

from env.sokoban import State, SokobanEnv, BFSPlanner
from env.alfworld import StateAlfworld

ORTOOLS_AVAILABLE = True
try:
    from ortools.sat.python import cp_model
except Exception:
    ORTOOLS_AVAILABLE = False
    cp_model = None  # type: ignore


def rec_recommend(
    env: SokobanEnv,
    planner: BFSPlanner,
    rng: random.Random,
    st: State,
    budget_remaining: int,
    delta: float,
) -> str:
    avail = env.available_actions(st)
    if not avail:
        return "U"

    plan = planner.shortest_plan(st)
    base = plan[0] if (plan is not None and len(plan) > 0) else rng.choice(avail)

    if rng.random() < delta:
        viable = set(planner.viable_actions(st, budget_remaining))
        non_viable = [a for a in avail if a not in viable]
        if non_viable:
            return rng.choice(non_viable)
        others = [a for a in avail if a != base]
        return rng.choice(others) if others else base

    return base

def select_path_ilp(
    start: Union[State, StateAlfworld],
    goals: Set[Union[State, StateAlfworld]],
    edges: List[Tuple[Union[State, StateAlfworld], str, Union[State, StateAlfworld]]],
    node_score: Dict[Union[State, StateAlfworld], float],
    budget: int,
    time_limit_sec: float = 5.0,
    score_scale: int = 1000,
    discount_factor: float = 1.0,
    return_states: bool = False,
    max_retry: int = 5,  # 새로운 파라미터 추가
    edge_time: Optional[
        Dict[Tuple[Union[State, StateAlfworld], str, Union[State, StateAlfworld]], float]
    ] = None,
    edge_cost: Optional[
        Dict[Tuple[Union[State, StateAlfworld], str, Union[State, StateAlfworld]], float]
    ] = None,
    time_budget: Optional[float] = None,
    cost_budget: Optional[float] = None,
    resource_scale: int = 1000,
) -> Optional[List[str]]:
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools not available. Install with: pip install ortools")
    assert cp_model is not None

    if budget <= 0 or not edges or not goals:
        return None

    # Practical cap for tractability: variables are |E| * L_max.
    # Use budget as a natural upper bound, and cap it to keep CP-SAT fast.
    L_CAP = 200  # adjust if needed
    L_max_base = min(budget, L_CAP)
    if L_max_base <= 0:
        return None

    # Build node set and edge list; add STOP self-loop for each goal to allow "stay at goal".
    E = list(edges)
    nodes: Set[State] = set([start]) | set(goals)
    for u, a, v in edges:
        nodes.add(u)
        nodes.add(v)

    # Add STOP loops (if not already present)
    stop_edges = []
    existing = set((u, a, v) for (u, a, v) in E)
    for g in goals:
        nodes.add(g)
        tup = (g, "STOP", g)
        if tup not in existing:
            stop_edges.append(tup)
    E.extend(stop_edges)

    m = len(E)
    edge_time = edge_time or {}
    edge_cost = edge_cost or {}
    # Convert per-edge resources to scaled nonnegative ints for CP-SAT.
    time_coeff: List[int] = []
    cost_coeff: List[int] = []
    for e in E:
        # Default per-edge resource is 1 when not provided, matching
        # "one-step-per-edge" intuition used in other tasks.
        # Keep STOP edges at 0 to avoid charging padding after reaching goal.
        if e[1] == "STOP":
            t_raw = 0.0
            c_raw = 0.0
        else:
            t_raw = edge_time[e] if e in edge_time else 1.0
            c_raw = edge_cost[e] if e in edge_cost else 1.0
        t_val = max(0.0, float(t_raw))
        c_val = max(0.0, float(c_raw))
        time_coeff.append(int(round(t_val * resource_scale)))
        cost_coeff.append(int(round(c_val * resource_scale)))

    # Precompute outgoing/incoming indices for constraints
    out_edges: Dict[State, List[int]] = defaultdict(list)
    in_edges: Dict[State, List[int]] = defaultdict(list)
    for i, (u, a, v) in enumerate(E):
        out_edges[u].append(i)
        in_edges[v].append(i)

    goal_list = [g for g in goals if g in nodes]
    if not goal_list:
        return None

    # L_max를 점진적으로 늘려가며 재시도
    for retry in range(max_retry + 1):
        L_max = L_max_base + retry
        if L_max > L_CAP:
            break

        model = cp_model.CpModel()

        # x[i, l] = whether choose edge i at step l
        x: Dict[Tuple[int, int], cp_model.IntVar] = {}
        for i in range(m):
            for l in range(L_max):
                x[(i, l)] = model.NewBoolVar(f"x_{i}_{l}")

        # y[u, l] = whether the walk is at node u at step l
        y: Dict[Tuple[State, int], cp_model.IntVar] = {}
        for u in nodes:
            for l in range(L_max + 1):
                y[(u, l)] = model.NewBoolVar(f"y_{hash(u)}_{l}")

        def sum_x(idxs: List[int], l: int):
            if not idxs:
                return 0
            return cp_model.LinearExpr.Sum([x[(i, l)] for i in idxs])

        # Initial state: at start at step 0
        model.Add(y[(start, 0)] == 1)
        for u in nodes:
            if u != start:
                model.Add(y[(u, 0)] == 0)

        # Exactly one node occupied at each step
        for l in range(L_max + 1):
            model.Add(cp_model.LinearExpr.Sum([y[(u, l)] for u in nodes]) == 1)

        # Exactly one edge taken at each step
        for l in range(L_max):
            model.Add(cp_model.LinearExpr.Sum([x[(i, l)] for i in range(m)]) == 1)

        # Optional resource constraints over the full path.
        if time_budget is not None and math.isfinite(time_budget):
            time_limit = int(round(max(0.0, float(time_budget)) * resource_scale))
            time_terms = []
            for i in range(m):
                coeff = time_coeff[i]
                if coeff <= 0:
                    continue
                for l in range(L_max):
                    time_terms.append(coeff * x[(i, l)])
            model.Add((cp_model.LinearExpr.Sum(time_terms) if time_terms else 0) <= time_limit)

        if cost_budget is not None and math.isfinite(cost_budget):
            cost_limit = int(round(max(0.0, float(cost_budget)) * resource_scale))
            cost_terms = []
            for i in range(m):
                coeff = cost_coeff[i]
                if coeff <= 0:
                    continue
                for l in range(L_max):
                    cost_terms.append(coeff * x[(i, l)])
            model.Add((cp_model.LinearExpr.Sum(cost_terms) if cost_terms else 0) <= cost_limit)

        # Link y and x (flow across time):
        # If at node u at step l, must take exactly one outgoing edge from u at step l.
        for l in range(L_max):
            for u in nodes:
                model.Add(sum_x(out_edges.get(u, []), l) == y[(u, l)])

        # Next node occupancy is determined by the chosen edge targets
        for l in range(L_max):
            for u in nodes:
                model.Add(sum_x(in_edges.get(u, []), l) == y[(u, l + 1)])

        # Must be at a goal node at the final step (reaching early is handled by STOP)
        model.Add(cp_model.LinearExpr.Sum([y[(g, L_max)] for g in goal_list]) == 1)

        # Objective: maximize discounted sum of node scores along visited nodes (excluding step 0)
        obj_terms = []
        for l in range(1, L_max + 1):
            step_discount = discount_factor ** (l - 1)
            for u in nodes:
                if u == start:
                    continue
                coeff = int(round(node_score.get(u, 0.0) * step_discount * score_scale))
                if coeff != 0:
                    obj_terms.append(coeff * y[(u, l)])
        model.Maximize(cp_model.LinearExpr.Sum(obj_terms) if obj_terms else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
        solver.parameters.num_search_workers = 8  # Number of parallel workers
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Reconstruct actions step-by-step
            actions: List[str] = []
            states: List[State] = [start]
            cur = start
            for l in range(L_max):
                chosen_i = None
                for i in out_edges.get(cur, []):
                    if solver.Value(x[(i, l)]) == 1:
                        chosen_i = i
                        break
                if chosen_i is None:
                    # should not happen if constraints are consistent
                    return None
                u, a, v = E[chosen_i]
                actions.append(a)
                cur = v
                states.append(cur)
                # Optionally stop early when first reaching goal (ignore remaining STOP padding)
                if cur in goals:
                    break

            # Remove trailing STOPs if any
            while actions and actions[-1] == "STOP":
                actions.pop()
                if len(states) > 1:
                    states.pop()

            if not actions:
                return None
            if return_states:
                return actions, states  # type: ignore[return-value]
            return actions
        
        # 해를 못 찾으면 L_max를 늘려서 재시도
    # 모든 재시도 후에도 해를 못 찾음
    return None

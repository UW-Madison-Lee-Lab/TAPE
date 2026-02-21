from __future__ import annotations

from typing import Dict, List, Tuple, Set, Any
import random
import math

from env.sokoban import State, SokobanEnv, BFSPlanner
from utils.planning import rec_recommend, select_path_ilp, ORTOOLS_AVAILABLE


class OursGraphILPPolicy:
    def __init__(self, env: SokobanEnv, planner: BFSPlanner, delta: float, N: int, seed: int = 0):
        self.env = env
        self.planner = planner
        self.delta = float(delta)
        self.N = int(N)
        self.rng = random.Random(seed)

    def reset_episode(self, seed: int):
        self.rng.seed(seed)

    def _node_score(self, u: State) -> float:
        d = self.planner.dist_to_goal(u)
        if d is math.inf:
            return 0.0
        return 1.0 / (1.0 + float(d))

    def _sample_subplan(self, st: State, max_len: int) -> Tuple[List[State], List[str]]:
        traj = [st]
        acts: List[str] = []
        cur = st
        for _ in range(max_len):
            if self.env.is_goal(cur):
                break
            a = rec_recommend(self.env, self.planner, self.rng, cur, max_len, self.delta)
            nxt = self.env._apply(cur, a)
            if nxt is None:
                break
            acts.append(a)
            cur = nxt
            traj.append(cur)
        return traj, acts

    def _build_folded_graph(self, st: State, budget_remaining: int) -> Tuple[Set[State], List[Tuple[State, str, State]]]:
        V: Set[State] = set()
        E: List[Tuple[State, str, State]] = []
        for _ in range(self.N):
            traj, acts = self._sample_subplan(st, budget_remaining)
            for x in traj:
                V.add(x)
            for i, a in enumerate(acts):
                E.append((traj[i], a, traj[i + 1]))
        return V, E

    def act(self, st: State, budget_remaining: int) -> Tuple[str, Dict[str, Any]]:
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError("OR-Tools not available. Install with: pip install ortools")

        if budget_remaining <= 0:
            return "U", {"d_cur": 0, "V": 0, "E": 0, "path_len": 0, "used_ilp": False}

        V, E = self._build_folded_graph(st, budget_remaining)
        outA = set(a for (u, a, v) in E if u == st)
        d_cur = len(outA)

        goals = {u for u in V if self.env.is_goal(u)}
        if not goals or not E:
            tilde = rec_recommend(self.env, self.planner, self.rng, st, budget_remaining, self.delta)
            return tilde, {"d_cur": d_cur, "V": len(V), "E": len(E), "path_len": 0, "used_ilp": False}

        node_score = {u: self._node_score(u) for u in V}
        actions = select_path_ilp(
            start=st,
            goals=goals,
            edges=E,
            node_score=node_score,
            budget=budget_remaining,
            time_limit_sec=2.0,
            score_scale=1000,
        )
        if not actions:
            tilde = rec_recommend(self.env, self.planner, self.rng, st, budget_remaining, self.delta)
            return tilde, {"d_cur": d_cur, "V": len(V), "E": len(E), "path_len": 0, "used_ilp": True}

        return actions[0], {"d_cur": d_cur, "V": len(V), "E": len(E), "path_len": len(actions), "used_ilp": True}

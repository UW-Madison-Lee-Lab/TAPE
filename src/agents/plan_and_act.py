from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import random

from env.sokoban import State, SokobanEnv, BFSPlanner
from utils.planning import rec_recommend


class PlanAndActPolicy:
    """
    Global plan generation also has planning error delta_plan.
    Execution:
      - if aligned and with prob p_follow: forced execute planned action (no exec noise)
      - else: fallback ReAct-style (delta_react + eta)
    """
    def __init__(
        self,
        env: SokobanEnv,
        planner: BFSPlanner,
        delta_plan: float,
        delta_react: float,
        eta: float,
        p_follow: float,
        seed: int = 0,
        max_plan_len: Optional[int] = None,
    ):
        self.env = env
        self.planner = planner
        self.delta_plan = float(delta_plan)
        self.delta_react = float(delta_react)
        self.eta = float(eta)
        self.p_follow = float(p_follow)
        self.rng = random.Random(seed)
        self.max_plan_len = max_plan_len

        self._plan_actions: Optional[List[str]] = None
        self._plan_states: Optional[List[State]] = None
        self._plan_stats: Dict[str, Any] = {}

    def reset_episode(self, seed: int):
        self.rng.seed(seed)
        self._plan_actions = None
        self._plan_states = None
        self._plan_stats = {}

    def _build_global_plan(self, st0: State, B_total: int):
        L = B_total if self.max_plan_len is None else min(B_total, self.max_plan_len)

        states = [st0]
        actions: List[str] = []
        cur = st0

        n_steps = 0
        n_nonviable = 0

        for t in range(L):
            if self.env.is_goal(cur):
                break

            a_bar = rec_recommend(self.env, self.planner, self.rng, cur, budget_remaining=(L - t), delta=self.delta_plan)

            viable = set(self.planner.viable_actions(cur, budget_remaining=(L - t)))
            if a_bar not in viable:
                n_nonviable += 1

            nxt = self.env._apply(cur, a_bar)
            if nxt is None:
                break

            actions.append(a_bar)
            cur = nxt
            states.append(cur)
            n_steps += 1

        self._plan_actions = actions
        self._plan_states = states
        self._plan_stats = {
            "plan_len": n_steps,
            "plan_nonviable": n_nonviable,
            "delta_plan_hat": (n_nonviable / n_steps) if n_steps > 0 else float("nan"),
        }

    def _is_aligned(self, st: State, t: int) -> bool:
        if self._plan_states is None:
            return False
        return 0 <= t < len(self._plan_states) and self._plan_states[t] == st

    def act(self, st: State, t: int, st0: State, budget_remaining: int, B_total: int) -> Tuple[str, str, bool]:
        if self._plan_actions is None or self._plan_states is None:
            self._build_global_plan(st0, B_total=B_total)

        aligned = self._is_aligned(st, t)

        if aligned and self._plan_actions is not None and t < len(self._plan_actions):
            if self.rng.random() < self.p_follow:
                a_bar = self._plan_actions[t]
                return a_bar, a_bar, True

        intended = rec_recommend(self.env, self.planner, self.rng, st, budget_remaining, self.delta_react)
        executed = intended
        if self.rng.random() < self.eta:
            avail = self.env.available_actions(st)
            others = [x for x in avail if x != intended]
            if others:
                executed = self.rng.choice(others)
        return executed, intended, False

    def get_plan_stats(self) -> Dict[str, Any]:
        return dict(self._plan_stats)

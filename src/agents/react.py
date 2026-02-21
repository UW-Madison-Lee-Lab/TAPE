from __future__ import annotations

from typing import Tuple
import random

from env.sokoban import State, SokobanEnv, BFSPlanner
from utils.planning import rec_recommend


class ReActPolicy:
    def __init__(self, env: SokobanEnv, planner: BFSPlanner, delta: float, eta: float, seed: int = 0):
        self.env = env
        self.planner = planner
        self.delta = float(delta)
        self.eta = float(eta)
        self.rng = random.Random(seed)

    def reset_episode(self, seed: int):
        self.rng.seed(seed)

    def act(self, st: State, budget_remaining: int) -> Tuple[str, str]:
        intended = rec_recommend(self.env, self.planner, self.rng, st, budget_remaining, self.delta)
        executed = intended
        if self.rng.random() < self.eta:
            avail = self.env.available_actions(st)
            others = [x for x in avail if x != intended]
            if others:
                executed = self.rng.choice(others)
        return executed, intended

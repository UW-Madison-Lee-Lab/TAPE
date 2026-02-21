from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Iterable
import math


ACTIONS = ["U", "D", "L", "R"]
DIR = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass(frozen=True)
class State:
    player: Tuple[int, int]
    boxes: Tuple[Tuple[int, int]]


class SokobanEnv:
    """
    Grid legend:
      # wall
      . goal
      $ box
      @ player
      * box on goal
      + player on goal
      (space) empty
    """
    def __init__(self, grid_str: str):
        self._parse(grid_str)
        self.reset()

    def _parse(self, s: str):
        lines = [list(x) for x in s.strip("\n").split("\n")]
        if not lines:
            raise ValueError("Empty map.")
        W = len(lines[0])
        if any(len(row) != W for row in lines):
            raise ValueError("Map must be rectangular.")
        self.H, self.W = len(lines), W

        self.walls: Set[Tuple[int, int]] = set()
        self.goals: Set[Tuple[int, int]] = set()
        boxes: List[Tuple[int, int]] = []
        player: Optional[Tuple[int, int]] = None

        for r in range(self.H):
            for c in range(self.W):
                ch = lines[r][c]
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == ".":
                    self.goals.add((r, c))
                elif ch == "@":
                    player = (r, c)
                elif ch == "$":
                    boxes.append((r, c))
                elif ch == "*":
                    boxes.append((r, c))
                    self.goals.add((r, c))
                elif ch == "+":
                    player = (r, c)
                    self.goals.add((r, c))
                elif ch == " ":
                    pass
                else:
                    raise ValueError(f"Unknown char '{ch}' in map.")

        if player is None:
            raise ValueError("No player '@' or '+' in map.")
        if len(boxes) == 0:
            raise ValueError("No boxes '$' or '*' in map.")
        if len(self.goals) == 0:
            raise ValueError("No goals '.' or '*' or '+' in map.")

        self.start_player = player
        self.start_boxes = tuple(sorted(boxes))

    def reset(self) -> State:
        self.state = State(self.start_player, self.start_boxes)
        return self.state

    def is_goal(self, st: State) -> bool:
        return set(st.boxes) == set(self.goals)

    def available_actions(self, st: State) -> List[str]:
        pr, pc = st.player
        avail = []
        for a in ACTIONS:
            dr, dc = DIR[a]
            nr, nc = pr + dr, pc + dc
            if (nr, nc) in self.walls:
                continue
            avail.append(a)
        return avail

    def _apply(self, st: State, a: str) -> Optional[State]:
        if a not in ACTIONS:
            return None
        dr, dc = DIR[a]
        pr, pc = st.player
        nr, nc = pr + dr, pc + dc
        if (nr, nc) in self.walls:
            return None

        boxes = set(st.boxes)
        if (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            if (br, bc) in self.walls or (br, bc) in boxes:
                return None
            boxes.remove((nr, nc))
            boxes.add((br, bc))
            return State((nr, nc), tuple(sorted(boxes)))
        return State((nr, nc), st.boxes)

    def step(self, a: str) -> Tuple[State, str, Dict]:
        nxt = self._apply(self.state, a)
        if nxt is None:
            return self.state, "fail", {}
        self.state = nxt
        return self.state, "success", {}


class BFSPlanner:
    def __init__(self, env: SokobanEnv, max_nodes: int = 200000):
        self.env = env
        self.max_nodes = max_nodes
        self._plan_cache: Dict[State, Optional[List[str]]] = {}
        self._dist_cache: Dict[State, float] = {}

    def neighbors(self, st: State) -> Iterable[Tuple[str, State]]:
        for a in ACTIONS:
            nxt = self.env._apply(st, a)
            if nxt is not None:
                yield a, nxt

    def shortest_plan(self, st: State) -> Optional[List[str]]:
        if st in self._plan_cache:
            return self._plan_cache[st]

        if self.env.is_goal(st):
            self._plan_cache[st] = []
            self._dist_cache[st] = 0.0
            return []

        q = deque([st])
        parent: Dict[State, Tuple[Optional[State], Optional[str]]] = {st: (None, None)}
        dist: Dict[State, int] = {st: 0}
        nodes = 0
        found: Optional[State] = None

        while q:
            cur = q.popleft()
            nodes += 1
            if nodes > self.max_nodes:
                break
            for a, nxt in self.neighbors(cur):
                if nxt in dist:
                    continue
                dist[nxt] = dist[cur] + 1
                parent[nxt] = (cur, a)
                if self.env.is_goal(nxt):
                    found = nxt
                    q.clear()
                    break
                q.append(nxt)

        if found is None:
            self._plan_cache[st] = None
            self._dist_cache[st] = math.inf
            return None

        acts: List[str] = []
        x = found
        while True:
            px, pa = parent[x]
            if px is None:
                break
            acts.append(pa)  # type: ignore
            x = px
        acts.reverse()

        self._plan_cache[st] = acts
        self._dist_cache[st] = float(len(acts))
        return acts

    def dist_to_goal(self, st: State) -> float:
        if st in self._dist_cache:
            return self._dist_cache[st]
        _ = self.shortest_plan(st)
        return self._dist_cache.get(st, math.inf)

    def viable_actions(self, st: State, budget_remaining: Optional[int] = None) -> List[str]:
        viable = []
        for a, nxt in self.neighbors(st):
            d = self.dist_to_goal(nxt)
            if d is math.inf:
                continue
            if budget_remaining is not None and d > (budget_remaining - 1):
                continue
            viable.append(a)
        return viable

from __future__ import annotations

from typing import List, Tuple, Optional, Set, Dict, Any, Iterable
import random
import os
import json
import math

from .sokoban import ACTIONS, DIR, State, SokobanEnv, BFSPlanner


def _rand_empty_cells(
    rng: random.Random,
    H: int,
    W: int,
    walls: Set[Tuple[int, int]],
    banned: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    cells = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if (r, c) in walls:
                continue
            if (r, c) in banned:
                continue
            cells.append((r, c))
    rng.shuffle(cells)
    return cells


def make_room_walls(H: int, W: int, rng: random.Random, internal_wall_prob: float) -> Set[Tuple[int, int]]:
    walls: Set[Tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                walls.add((r, c))
    if internal_wall_prob > 0:
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if rng.random() < internal_wall_prob:
                    walls.add((r, c))
    return walls


def render_map(H: int, W: int, walls: Set[Tuple[int, int]], goals: List[Tuple[int, int]], st: State) -> str:
    grid = [[" " for _ in range(W)] for _ in range(H)]
    for (r, c) in walls:
        grid[r][c] = "#"
    goal_set = set(goals)
    for (r, c) in goals:
        if grid[r][c] != "#":
            grid[r][c] = "."
    boxes = set(st.boxes)
    for (r, c) in boxes:
        if (r, c) in goal_set:
            grid[r][c] = "*"
        else:
            grid[r][c] = "$"
    pr, pc = st.player
    if (pr, pc) in goal_set:
        grid[pr][pc] = "+"
    else:
        grid[pr][pc] = "@"
    return "\n".join("".join(row) for row in grid)


def reverse_neighbors(env: SokobanEnv, st: State) -> Iterable[State]:
    """
    Reverse move: player moves; can 'pull' a box from behind into player's current cell.
    Walls/goals are fixed in env. Boxes/player are in st.
    """
    pr, pc = st.player
    boxes = set(st.boxes)

    for a in ACTIONS:
        dr, dc = DIR[a]
        npr, npc = pr + dr, pc + dc
        if (npr, npc) in env.walls:
            continue
        if (npr, npc) in boxes:
            continue

        yield State((npr, npc), st.boxes)

        bpos = (pr - dr, pc - dc)
        if bpos in boxes:
            if (pr, pc) in env.walls:
                continue
            if (pr, pc) in boxes:
                continue
            new_boxes = set(boxes)
            new_boxes.remove(bpos)
            new_boxes.add((pr, pc))
            yield State((npr, npc), tuple(sorted(new_boxes)))


def generate_sokoban_reverse(
    H: int,
    W: int,
    num_boxes: int,
    reverse_steps: int,
    internal_wall_prob: float,
    seed: int,
) -> str:
    rng = random.Random(seed)

    walls = make_room_walls(H, W, rng, internal_wall_prob)

    goals: List[Tuple[int, int]] = []
    banned: Set[Tuple[int, int]] = set()
    goal_cells = _rand_empty_cells(rng, H, W, walls, banned)
    for _ in range(num_boxes):
        if not goal_cells:
            break
        g = goal_cells.pop()
        goals.append(g)
        banned.add(g)
    if len(goals) != num_boxes:
        raise RuntimeError("Failed to place goals (try larger grid or lower wall prob).")

    boxes = tuple(sorted(goals))

    empties = _rand_empty_cells(rng, H, W, walls, set(goals))
    if not empties:
        raise RuntimeError("Failed to place player.")
    player = empties[0]

    solved_state = State(player=player, boxes=boxes)
    solved_map = render_map(H, W, walls, goals, solved_state)
    env = SokobanEnv(solved_map)
    st = env.reset()

    for _ in range(reverse_steps):
        nbrs = list(reverse_neighbors(env, st))
        if not nbrs:
            break
        st = rng.choice(nbrs)

    final_map = render_map(H, W, env.walls, sorted(list(env.goals)), st)
    return final_map


def compute_T_star(map_str: str, max_nodes: int) -> Optional[int]:
    try:
        env = SokobanEnv(map_str)
    except Exception:
        return None
    planner = BFSPlanner(env, max_nodes=max_nodes)
    s0 = env.reset()
    d = planner.dist_to_goal(s0)
    if d is math.inf:
        return None
    return int(d)


def build_dataset_by_Tstar(
    out_path: str,
    num_maps_per_bucket: int,
    T_targets: List[int],
    T_tol: int,
    H: int,
    W: int,
    num_boxes: int,
    reverse_steps_min: int,
    reverse_steps_max: int,
    internal_wall_prob: float,
    max_nodes: int,
    seed: int,
    max_tries: int = 200000,
):
    rng = random.Random(seed)
    buckets: Dict[int, List[Dict[str, Any]]] = {t: [] for t in T_targets}

    tries = 0
    while tries < max_tries:
        done = all(len(buckets[t]) >= num_maps_per_bucket for t in T_targets)
        if done:
            break

        tries += 1
        rev_steps = rng.randint(reverse_steps_min, reverse_steps_max)
        mp_seed = rng.randint(0, 10**9)

        try:
            mp = generate_sokoban_reverse(
                H=H,
                W=W,
                num_boxes=num_boxes,
                reverse_steps=rev_steps,
                internal_wall_prob=internal_wall_prob,
                seed=mp_seed,
            )
        except Exception:
            continue

        Tstar = compute_T_star(mp, max_nodes=max_nodes)
        if Tstar is None:
            continue

        for t in T_targets:
            if len(buckets[t]) >= num_maps_per_bucket:
                continue
            if abs(Tstar - t) <= T_tol:
                buckets[t].append({
                    "map": mp,
                    "T_star": Tstar,
                    "reverse_steps": rev_steps,
                    "seed": mp_seed,
                    "H": H,
                    "W": W,
                    "num_boxes": num_boxes,
                    "internal_wall_prob": internal_wall_prob,
                })
                break

    dataset: List[Dict[str, Any]] = []
    for t in T_targets:
        dataset.extend(buckets[t])

    meta = {
        "num_maps_per_bucket": num_maps_per_bucket,
        "T_targets": T_targets,
        "T_tol": T_tol,
        "H": H,
        "W": W,
        "num_boxes": num_boxes,
        "reverse_steps_min": reverse_steps_min,
        "reverse_steps_max": reverse_steps_max,
        "internal_wall_prob": internal_wall_prob,
        "max_nodes": max_nodes,
        "seed": seed,
        "tries": tries,
        "count": len(dataset),
        "bucket_counts": {t: len(buckets[t]) for t in T_targets},
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "data": dataset}, f, indent=2, ensure_ascii=False)

    print(f"[Saved dataset] {out_path}")
    print("Bucket counts:", meta["bucket_counts"])
    if tries >= max_tries:
        print("[Warning] Reached max_tries; some buckets may be underfilled.")

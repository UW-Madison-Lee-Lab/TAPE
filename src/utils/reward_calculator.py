"""
Reward function calculator for DAG-based tool selection.

Computes V(node) = expected reward from each node using backward Bellman equation:
    V(goal) = 1.0
    V(node) = max_{edge ∈ outgoing(node)} [ quality(edge) × V(next_node) ]

This value can be used as node_score in select_path_ilp() for optimal path finding.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import json
import inspect

import tools.arithmetic_tools as arithmetic_tools_module


def _build_tool_quality_registry() -> Dict[str, float]:
    """
    Automatically build TOOL_QUALITY_REGISTRY by scanning all classes
    in arithmetic_tools.py that have 'name' and 'output_quality' attributes.
    """
    registry = {}
    
    for name, obj in inspect.getmembers(arithmetic_tools_module, inspect.isclass):
        # Check if class has required attributes
        if hasattr(obj, 'name') and hasattr(obj, 'output_quality'):
            tool_name = getattr(obj, 'name', None)
            output_quality = getattr(obj, 'output_quality', None)
            
            if tool_name and output_quality is not None:
                registry[tool_name] = output_quality
    
    return registry


# Build registry automatically from arithmetic_tools.py
TOOL_QUALITY_REGISTRY: Dict[str, float] = _build_tool_quality_registry()


def get_tool_quality(tool_name: str) -> float:
    """
    Get output_quality for a tool by name.
    Returns 1.0 (no noise) for unknown tools.
    """
    return TOOL_QUALITY_REGISTRY.get(tool_name, 1.0)


@dataclass
class DAGEdge:
    """Represents an edge (action) in the DAG."""
    from_node: str
    to_node: str
    action: str
    quality: float = 1.0
    cost: float = 0.0


class RewardCalculator:
    """
    Computes V(node) for a DAG with noisy tool actions.
    
    Graph structure (from JSON):
    {
        "nodes": [{"id": "node1", "is_goal": false, ...}, ...],
        "edges": [{"from": "node1", "to": "node2", "action": "add_old"}, ...]
    }
    
    Usage:
        calc = RewardCalculator(graph)
        node_scores = calc.compute_all_node_values()
        # Pass node_scores to select_path_ilp()
    """
    
    def __init__(self, graph: Dict):
        """
        Initialize from graph dictionary.
        
        Args:
            graph: Dictionary with "nodes" and "edges" keys
        """
        self.nodes: Set[str] = set()
        self.edges: List[DAGEdge] = []
        self.incoming: Dict[str, List[DAGEdge]] = {}
        self.outgoing: Dict[str, List[DAGEdge]] = {}
        self.goal_nodes: Set[str] = set()
        self.start_nodes: Set[str] = set()
        
        self._parse_graph(graph)
    
    def _parse_graph(self, graph: Dict):
        """Parse graph dictionary into internal structures."""
        # Parse nodes
        for node_data in graph.get("nodes", []):
            node_id = node_data.get("id") or f"node{len(self.nodes) + 1}"
            self.nodes.add(node_id)
            self.incoming[node_id] = []
            self.outgoing[node_id] = []
            
            if node_data.get("is_goal", False):
                self.goal_nodes.add(node_id)
        
        # Parse edges
        for edge_data in graph.get("edges", []):
            action = edge_data.get("action", "")
            quality = get_tool_quality(action)
            cost = edge_data.get("cost", 0.0)
            
            from_node = edge_data["from"]
            to_node = edge_data["to"]
            
            # Ensure nodes exist
            for n in [from_node, to_node]:
                if n not in self.nodes:
                    self.nodes.add(n)
                    self.incoming[n] = []
                    self.outgoing[n] = []
            
            edge = DAGEdge(
                from_node=from_node,
                to_node=to_node,
                action=action,
                quality=quality,
                cost=cost,
            )
            self.edges.append(edge)
            self.incoming[to_node].append(edge)
            self.outgoing[from_node].append(edge)
        
        # Find start nodes (no incoming edges)
        for node_id in self.nodes:
            if not self.incoming[node_id]:
                self.start_nodes.add(node_id)
    
    def compute_node_value(
        self,
        node_id: str,
        memo: Optional[Dict[str, float]] = None,
        aggregation: str = "max",
    ) -> float:
        """
        Compute V(node) = expected reward from this node (backward Bellman).
        
        Bellman equation:
            V(goal) = 1.0
            V(node) = agg_{edge ∈ outgoing(node)} [ quality(edge) × V(next_node) ]
        
        Args:
            node_id: Target node ID
            memo: Memoization cache
            aggregation: "max" for optimal policy, "sum" for expected value over all paths
            
        Returns:
            V(node) = expected reward from this node
        """
        if memo is None:
            memo = {}
        
        if node_id in memo:
            return memo[node_id]
        
        # Base case: goal node
        if node_id in self.goal_nodes:
            memo[node_id] = 1.0
            return 1.0
        
        # Get outgoing edges
        outgoing_edges = self.outgoing.get(node_id, [])
        
        # Dead end: no path to goal
        if not outgoing_edges:
            memo[node_id] = 0.0
            return 0.0
        
        # Compute value for each outgoing edge
        edge_values = []
        for edge in outgoing_edges:
            next_value = self.compute_node_value(edge.to_node, memo, aggregation)
            edge_value = edge.quality * next_value
            edge_values.append(edge_value)
        
        # Aggregate: max (optimal policy) or sum/mean (stochastic)
        if aggregation == "max":
            node_value = max(edge_values)
        elif aggregation == "sum":
            node_value = sum(edge_values)
        elif aggregation == "mean":
            node_value = sum(edge_values) / len(edge_values)
        elif aggregation == "or":
            # OR: any edge succeeding is enough
            # P(success) = 1 - ∏(1 - p_i)
            fail_prob = 1.0
            for v in edge_values:
                fail_prob *= (1.0 - v)
            node_value = 1.0 - fail_prob
        else:
            node_value = max(edge_values)
        
        memo[node_id] = node_value
        return node_value
    
    def compute_all_node_values(self, aggregation: str = "max") -> Dict[str, float]:
        """
        Compute V(node) for all nodes in the graph.
        
        Args:
            aggregation: "max", "sum", "mean", or "or"
            
        Returns:
            Dictionary mapping node_id -> V(node)
        """
        memo: Dict[str, float] = {}
        for node_id in self.nodes:
            self.compute_node_value(node_id, memo, aggregation)
        return memo
    
    def compute_expected_reward(self, aggregation: str = "max") -> float:
        """
        Compute E[Reward] = V(start).
        
        Args:
            aggregation: "max", "sum", "mean", or "or"
            
        Returns:
            Expected reward from start node(s)
        """
        if not self.start_nodes:
            return 0.0
        
        memo = self.compute_all_node_values(aggregation)
        
        start_values = [memo.get(s, 0.0) for s in self.start_nodes]
        return max(start_values) if start_values else 0.0
    
    def compute_path_probability(self, path: List[str]) -> float:
        """
        Compute P(success) for a specific path (sequence of actions).
        
        Args:
            path: List of action names in order
            
        Returns:
            P(success) = ∏ quality(action)
        """
        prob = 1.0
        for action in path:
            prob *= get_tool_quality(action)
        return prob
    
    def get_optimal_path(self) -> Tuple[List[str], float]:
        """
        Get the optimal path (sequence of actions) using greedy selection.
        
        Returns:
            (actions, expected_reward)
        """
        if not self.start_nodes:
            return [], 0.0
        
        memo = self.compute_all_node_values(aggregation="max")
        
        # Start from the best start node
        current = max(self.start_nodes, key=lambda s: memo.get(s, 0.0))
        actions = []
        
        while current not in self.goal_nodes:
            outgoing = self.outgoing.get(current, [])
            if not outgoing:
                break  # Dead end
            
            # Pick edge with highest expected value
            best_edge = max(
                outgoing,
                key=lambda e: e.quality * memo.get(e.to_node, 0.0)
            )
            actions.append(best_edge.action)
            current = best_edge.to_node
        
        return actions, self.compute_path_probability(actions)


def compute_node_scores(graph: Dict, aggregation: str = "max") -> Dict[str, float]:
    """
    Convenience function to compute V(node) for all nodes.
    
    Args:
        graph: Dictionary with "nodes" and "edges" keys
        aggregation: "max", "sum", "mean", or "or"
        
    Returns:
        Dictionary mapping node_id -> V(node)
    """
    calc = RewardCalculator(graph)
    return calc.compute_all_node_values(aggregation)


def compute_expected_reward(graph: Dict, aggregation: str = "max") -> float:
    """
    Convenience function to compute E[Reward] from a graph.
    """
    calc = RewardCalculator(graph)
    return calc.compute_expected_reward(aggregation)


def compute_reward_from_json(filepath: str, aggregation: str = "max") -> float:
    """
    Compute E[Reward] from a JSON file.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    
    graph = data.get("graph", data)
    return compute_expected_reward(graph, aggregation)


def compute_scores_for_ilp(
    current_node: str,
    graph: Dict,
    goal_nodes: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    현재 node에서 ILP에 넘길 node_score 계산.
    
    - 현재 node에서 바로 연결된 다음 node들: quality(edge) × V(next_node)
    - goal nodes: 1.0
    - 나머지: 0.0
    
    Args:
        current_node: 현재 위치한 node ID
        graph: DAG graph dictionary
        goal_nodes: Goal node IDs (None이면 graph에서 자동 탐지)
        
    Returns:
        Dict[node_id, score] for ILP
    """
    calc = RewardCalculator(graph)
    
    # Goal nodes 설정
    if goal_nodes is None:
        goal_nodes = calc.goal_nodes
    
    # 전체 V(node) 계산 (backward from goal)
    V = calc.compute_all_node_values()
    
    # 모든 node 0으로 초기화
    scores = {node: 0.0 for node in calc.nodes}
    
    # 현재 node에서 바로 연결된 다음 node들만 score 계산
    for edge in calc.outgoing.get(current_node, []):
        next_node = edge.to_node
        edge_value = edge.quality * V.get(next_node, 0.0)
        scores[next_node] = edge_value
    
    # goal은 1.0
    for goal in goal_nodes:
        scores[goal] = 1.0
    
    return scores


# ----- Example usage -----

if __name__ == "__main__":
    
    # =========================================================
    # Example 1: Simple Sequential Path
    # =========================================================
    print("=" * 60)
    print("Example 1: Simple Sequential Path")
    print("=" * 60)
    print()
    print("  [start] --add_old(0.95)--> [n1] --mult_verified(1.0)--> [goal]")
    print()
    
    sequential_graph = {
        "nodes": [
            {"id": "start", "is_goal": False},
            {"id": "n1", "is_goal": False},
            {"id": "goal", "is_goal": True},
        ],
        "edges": [
            {"from": "start", "to": "n1", "action": "add_old"},
            {"from": "n1", "to": "goal", "action": "multiply_verified"},
        ],
    }
    
    calc1 = RewardCalculator(sequential_graph)
    scores1 = calc1.compute_all_node_values()
    
    print("  V(node) values:")
    for node, value in sorted(scores1.items()):
        print(f"    V({node}) = {value:.4f}")
    print()
    print(f"  E[Reward] = V(start) = {calc1.compute_expected_reward():.4f}")
    print(f"  Expected: 0.95 × 1.0 = 0.95")
    print()
    
    
    # =========================================================
    # Example 2: Diamond DAG with Choice
    # =========================================================
    print("=" * 60)
    print("Example 2: Diamond DAG with Choice")
    print("=" * 60)
    print()
    print("                 ┌── add_old(0.95) ──> [n1] ──mult_verified(1.0)──┐")
    print("  [start] ───────┤                                               ├──> [goal]")
    print("                 └── add_verylight(0.80) ──> [n2] ──mult_old(0.95)┘")
    print()
    
    diamond_graph = {
        "nodes": [
            {"id": "start", "is_goal": False},
            {"id": "n1", "is_goal": False},
            {"id": "n2", "is_goal": False},
            {"id": "goal", "is_goal": True},
        ],
        "edges": [
            {"from": "start", "to": "n1", "action": "add_old"},
            {"from": "start", "to": "n2", "action": "add_verylightweight"},
            {"from": "n1", "to": "goal", "action": "multiply_verified"},
            {"from": "n2", "to": "goal", "action": "multiply_old"},
        ],
    }
    
    calc2 = RewardCalculator(diamond_graph)
    scores2 = calc2.compute_all_node_values()
    
    print("  V(node) values (max aggregation = optimal policy):")
    for node, value in sorted(scores2.items()):
        print(f"    V({node}) = {value:.4f}")
    print()
    print("  Calculation:")
    print("    V(goal) = 1.0")
    print("    V(n1) = 1.0 × 1.0 = 1.0")
    print("    V(n2) = 0.95 × 1.0 = 0.95")
    print("    V(start) = max(0.95 × V(n1), 0.80 × V(n2))")
    print("             = max(0.95 × 1.0, 0.80 × 0.95)")
    print("             = max(0.95, 0.76) = 0.95")
    print()
    print(f"  E[Reward] = {calc2.compute_expected_reward():.4f}")
    print()
    
    path, prob = calc2.get_optimal_path()
    print(f"  Optimal path: {path}")
    print(f"  Path probability: {prob:.4f}")
    print()
    
    
    # =========================================================
    # Example 3: Complex DAG with Multiple Goals
    # =========================================================
    print("=" * 60)
    print("Example 3: Complex DAG with Multiple Goals")
    print("=" * 60)
    print()
    print("                   ┌── add_old(0.95) ──> [A] ──mult_old(0.95)──> [goal1]")
    print("  [start] ─────────┤")
    print("                   └── sub_verylight(0.80) ──> [B] ──div_verified(1.0)──> [goal2]")
    print()
    
    complex_graph = {
        "nodes": [
            {"id": "start", "is_goal": False},
            {"id": "A", "is_goal": False},
            {"id": "B", "is_goal": False},
            {"id": "goal1", "is_goal": True},
            {"id": "goal2", "is_goal": True},
        ],
        "edges": [
            {"from": "start", "to": "A", "action": "add_old"},
            {"from": "start", "to": "B", "action": "subtract_verylightweight"},
            {"from": "A", "to": "goal1", "action": "multiply_old"},
            {"from": "B", "to": "goal2", "action": "divide_verified"},
        ],
    }
    
    calc3 = RewardCalculator(complex_graph)
    scores3 = calc3.compute_all_node_values()
    
    print("  V(node) values:")
    for node, value in sorted(scores3.items()):
        print(f"    V({node}) = {value:.4f}")
    print()
    print("  Calculation:")
    print("    V(goal1) = V(goal2) = 1.0")
    print("    V(A) = 0.95 × 1.0 = 0.95")
    print("    V(B) = 1.0 × 1.0 = 1.0")
    print("    V(start) = max(0.95 × 0.95, 0.80 × 1.0)")
    print("             = max(0.9025, 0.80) = 0.9025")
    print()
    print(f"  E[Reward] = {calc3.compute_expected_reward():.4f}")
    print()
    
    path, prob = calc3.get_optimal_path()
    print(f"  Optimal path: {path}")
    print(f"  Path probability: {prob:.4f}")
    print()
    
    
    # =========================================================
    # Example 4: GSM-8K Style Arithmetic Chain
    # =========================================================
    print("=" * 60)
    print("Example 4: GSM-8K Style Arithmetic Chain")
    print("=" * 60)
    print()
    print("  Problem: ((a + b) × c - d) / e")
    print()
    print("  [start] ─add_old─> [s1] ─mult_verified─> [s2] ─sub_verylight─> [s3] ─div_old─> [goal]")
    print()
    
    gsm_graph = {
        "nodes": [
            {"id": "start", "is_goal": False},
            {"id": "s1", "is_goal": False},
            {"id": "s2", "is_goal": False},
            {"id": "s3", "is_goal": False},
            {"id": "goal", "is_goal": True},
        ],
        "edges": [
            {"from": "start", "to": "s1", "action": "add_old"},
            {"from": "s1", "to": "s2", "action": "multiply_verified"},
            {"from": "s2", "to": "s3", "action": "subtract_verylightweight"},
            {"from": "s3", "to": "goal", "action": "divide_old"},
        ],
    }
    
    calc4 = RewardCalculator(gsm_graph)
    scores4 = calc4.compute_all_node_values()
    
    print("  V(node) values (backward from goal):")
    for node in ["goal", "s3", "s2", "s1", "start"]:
        print(f"    V({node}) = {scores4[node]:.4f}")
    print()
    print("  Calculation:")
    print("    V(goal) = 1.0")
    print("    V(s3) = 0.95 × 1.0 = 0.95")
    print("    V(s2) = 0.80 × 0.95 = 0.76")
    print("    V(s1) = 1.0 × 0.76 = 0.76")
    print("    V(start) = 0.95 × 0.76 = 0.722")
    print()
    print(f"  E[Reward] = {calc4.compute_expected_reward():.4f}")
    print()
    
    
    # =========================================================
    # Summary
    # =========================================================
    print("=" * 60)
    print("Summary: node_scores for select_path_ilp()")
    print("=" * 60)
    print()
    print("  Usage:")
    print("    from utils.reward_calculator import compute_node_scores")
    print("    from utils.planning import select_path_ilp")
    print()
    print("    node_scores = compute_node_scores(graph)")
    print("    actions = select_path_ilp(start, goals, edges, node_scores, budget)")
    print()
#!/usr/bin/env python3
"""
Planning Error Analysis for Sokoban

이 스크립트는 report_pa.json 같은 분석 결과 파일을 읽어서,
각 스텝에서 inferred action을 실제로 수행한 후 BFSPlanner를 통해
shortest_plan 길이를 계산하여 planning error를 분석합니다.

성공 조건:
1. inferred_action이 valid하고, 해당 action 실행 후 shortest_plan 길이 <= steps_remaining - 1
2. inferred_action이 unknown인데, 어떤 action을 해도 shortest_plan 길이 > steps_remaining - 1

실패 조건:
1. inferred_action이 valid한데, 해당 action 실행 후 shortest_plan 길이 > steps_remaining - 1
2. inferred_action이 unknown인데, 어떤 action을 해도 shortest_plan 길이 <= steps_remaining - 1
"""

import json
import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import math

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.sokoban import SokobanEnv, BFSPlanner, ACTIONS


def calculate_se(rate: float, n: int) -> float:
    """
    이진 데이터(Bernoulli)에 대한 Standard Error를 계산합니다.
    SE = sqrt(p * (1 - p) / n)
    """
    if n <= 1:
        return 0.0
    return math.sqrt(rate * (1 - rate) / n)


@dataclass
class PlanningAnalysisResult:
    """단일 스텝의 planning 분석 결과"""
    file: str
    step: int
    steps_remaining: int
    inferred_action: Optional[str]
    actual_action: Optional[str]
    is_valid_action: bool  # inferred_action이 실행 가능한지
    new_state_shortest_plan_len: Optional[int]  # action 실행 후 shortest plan 길이
    planning_success: bool  # planning이 성공적인지
    planning_verdict: str  # 상세 판정 설명
    best_available_plan_len: Optional[int]  # 현재 상태에서 가장 짧은 plan 길이


def parse_observation_to_grid(observation: str) -> Tuple[str, Optional[str]]:
    """
    observation 문자열을 파싱하여 SokobanEnv용 grid 문자열을 생성합니다.
    
    Returns:
        (grid_str, error_message)
    """
    try:
        # observation 파싱
        walls = set()
        player = None
        boxes = []
        goals = []
        boxes_on_goal = []
        
        lines = observation.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('wall location:'):
                coords_str = line.replace('wall location:', '').strip()
                if coords_str.lower() != 'none':
                    coords = re.findall(r'\((\d+),\s*(\d+)\)', coords_str)
                    walls = {(int(x), int(y)) for x, y in coords}
            elif line.startswith('player location:'):
                coords_str = line.replace('player location:', '').strip()
                match = re.search(r'\((\d+),\s*(\d+)\)', coords_str)
                if match:
                    player = (int(match.group(1)), int(match.group(2)))
            elif line.startswith('box location:'):
                coords_str = line.replace('box location:', '').strip()
                if coords_str.lower() != 'none':
                    coords = re.findall(r'\((\d+),\s*(\d+)\)', coords_str)
                    boxes = [(int(x), int(y)) for x, y in coords]
            elif line.startswith('goal location:'):
                coords_str = line.replace('goal location:', '').strip()
                if coords_str.lower() != 'none':
                    coords = re.findall(r'\((\d+),\s*(\d+)\)', coords_str)
                    goals = [(int(x), int(y)) for x, y in coords]
            elif line.startswith('box on goal location:'):
                coords_str = line.replace('box on goal location:', '').strip()
                if coords_str.lower() != 'none':
                    coords = re.findall(r'\((\d+),\s*(\d+)\)', coords_str)
                    boxes_on_goal = [(int(x), int(y)) for x, y in coords]
        
        if player is None:
            return "", "No player found in observation"
        
        if not goals and not boxes_on_goal:
            return "", "No goals found in observation"
        
        # grid 크기 계산
        all_coords = list(walls) + [player] + boxes + goals + boxes_on_goal
        if not all_coords:
            return "", "No coordinates found"
        
        max_x = max(c[0] for c in all_coords) + 1
        max_y = max(c[1] for c in all_coords) + 1
        
        # grid 생성 (observation의 (x, y)는 (col, row) 형태)
        # SokobanEnv는 (row, col) 형태로 처리하므로 변환 필요
        grid = [[' ' for _ in range(max_x)] for _ in range(max_y)]
        
        # walls
        for x, y in walls:
            grid[y][x] = '#'
        
        # goals (box on goal이 아닌 것만)
        boxes_on_goal_set = set(boxes_on_goal)
        for x, y in goals:
            if (x, y) not in boxes_on_goal_set:
                grid[y][x] = '.'
        
        # boxes (box on goal이 아닌 것만)
        for x, y in boxes:
            if (x, y) in boxes_on_goal_set:
                grid[y][x] = '*'  # box on goal
            else:
                grid[y][x] = '$'
        
        # boxes on goal
        for x, y in boxes_on_goal:
            grid[y][x] = '*'
        
        # player
        px, py = player
        if (px, py) in set(goals) or (px, py) in boxes_on_goal_set:
            grid[py][px] = '+'  # player on goal
        else:
            grid[py][px] = '@'
        
        grid_str = '\n'.join(''.join(row) for row in grid)
        return grid_str, None
        
    except Exception as e:
        return "", f"Error parsing observation: {str(e)}"


def analyze_step(
    step_log: Dict[str, Any],
    max_nodes: int = 200000
) -> PlanningAnalysisResult:
    """
    단일 스텝을 분석합니다.
    """
    file = step_log.get('file', 'unknown')
    step = step_log.get('step', 0)
    steps_remaining = step_log.get('steps_remaining', 0)
    observation = step_log.get('observation', '')
    inferred_action = step_log.get('inferred_action')
    actual_action = step_log.get('actual_action')
    
    # observation 파싱
    grid_str, error = parse_observation_to_grid(observation)
    if error:
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=False,
            new_state_shortest_plan_len=None,
            planning_success=False,
            planning_verdict=f"Parse error: {error}",
            best_available_plan_len=None
        )
    
    # SokobanEnv 생성
    try:
        env = SokobanEnv(grid_str)
        planner = BFSPlanner(env, max_nodes=max_nodes)
    except Exception as e:
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=False,
            new_state_shortest_plan_len=None,
            planning_success=False,
            planning_verdict=f"Env creation error: {str(e)}",
            best_available_plan_len=None
        )
    
    current_state = env.state
    budget = steps_remaining - 1  # action 실행 후 남는 step
    
    # 현재 상태에서 가장 짧은 plan 길이
    current_shortest = planner.shortest_plan(current_state)
    current_shortest_len = len(current_shortest) if current_shortest is not None else None
    
    # inferred_action이 Unknown인 경우
    if inferred_action is None or inferred_action.upper() == 'UNKNOWN':
        # 모든 action에 대해 검사: 어떤 action을 해도 goal에 도달 불가능해야 성공
        any_viable = False
        for action in ACTIONS:
            next_state = env._apply(current_state, action)
            if next_state is not None:
                plan = planner.shortest_plan(next_state)
                if plan is not None and len(plan) <= budget:
                    any_viable = True
                    break
        
        if any_viable:
            # Unknown이지만 viable한 action이 존재 -> 실패
            return PlanningAnalysisResult(
                file=file,
                step=step,
                steps_remaining=steps_remaining,
                inferred_action=inferred_action,
                actual_action=actual_action,
                is_valid_action=False,
                new_state_shortest_plan_len=None,
                planning_success=False,
                planning_verdict="Unknown action but viable action exists",
                best_available_plan_len=current_shortest_len
            )
        else:
            # Unknown이고 실제로 모든 action이 불가능 -> 성공
            return PlanningAnalysisResult(
                file=file,
                step=step,
                steps_remaining=steps_remaining,
                inferred_action=inferred_action,
                actual_action=actual_action,
                is_valid_action=False,
                new_state_shortest_plan_len=None,
                planning_success=True,
                planning_verdict="Correctly identified no viable action",
                best_available_plan_len=current_shortest_len
            )
    
    # inferred_action 실행
    action = inferred_action.upper()
    if action not in ACTIONS:
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=False,
            new_state_shortest_plan_len=None,
            planning_success=False,
            planning_verdict=f"Invalid action: {inferred_action}",
            best_available_plan_len=current_shortest_len
        )
    
    next_state = env._apply(current_state, action)
    if next_state is None:
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=False,
            new_state_shortest_plan_len=None,
            planning_success=False,
            planning_verdict="Action blocked (wall or immovable box)",
            best_available_plan_len=current_shortest_len
        )
    
    # action 실행 후 shortest plan 계산
    new_plan = planner.shortest_plan(next_state)
    new_plan_len = len(new_plan) if new_plan is not None else None
    
    if new_plan is None:
        # goal에 도달 불가능
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=True,
            new_state_shortest_plan_len=None,
            planning_success=False,
            planning_verdict="Action leads to unsolvable state",
            best_available_plan_len=current_shortest_len
        )
    
    if new_plan_len <= budget:
        # 성공: goal에 도달 가능
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=True,
            new_state_shortest_plan_len=new_plan_len,
            planning_success=True,
            planning_verdict=f"Good action: {new_plan_len} steps needed, {budget} steps available",
            best_available_plan_len=current_shortest_len
        )
    else:
        # 실패: goal에 도달 가능하지만 budget 초과
        return PlanningAnalysisResult(
            file=file,
            step=step,
            steps_remaining=steps_remaining,
            inferred_action=inferred_action,
            actual_action=actual_action,
            is_valid_action=True,
            new_state_shortest_plan_len=new_plan_len,
            planning_success=False,
            planning_verdict=f"Bad action: {new_plan_len} steps needed > {budget} steps available",
            best_available_plan_len=current_shortest_len
        )


def analyze_report(
    report_path: str,
    max_nodes: int = 200000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    전체 report 파일을 분석합니다.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    all_step_logs = report.get('all_step_logs', [])
    
    if not all_step_logs:
        print("No step logs found in report")
        return {}
    
    results = []
    success_count = 0
    failure_count = 0
    parse_error_count = 0
    
    for step_log in all_step_logs:
        result = analyze_step(step_log, max_nodes=max_nodes)
        results.append(result)
        
        if 'error' in result.planning_verdict.lower():
            parse_error_count += 1
        elif result.planning_success:
            success_count += 1
        else:
            failure_count += 1
        
        if verbose:
            print(f"\n[{result.file}] Step {result.step}")
            print(f"  Steps remaining: {result.steps_remaining}")
            print(f"  Inferred action: {result.inferred_action}")
            print(f"  Actual action: {result.actual_action}")
            print(f"  Valid action: {result.is_valid_action}")
            print(f"  New state shortest plan: {result.new_state_shortest_plan_len}")
            print(f"  Planning success: {result.planning_success}")
            print(f"  Verdict: {result.planning_verdict}")
    
    total = len(results)
    valid_total = total - parse_error_count
    
    # 비율 및 Standard Error 계산
    success_rate = success_count / valid_total if valid_total > 0 else 0
    failure_rate = failure_count / valid_total if valid_total > 0 else 0
    success_se = calculate_se(success_rate, valid_total)
    failure_se = calculate_se(failure_rate, valid_total)
    
    summary = {
        "report_file": report_path,
        "total_steps": total,
        "parse_errors": parse_error_count,
        "valid_steps": valid_total,
        "planning_success": success_count,
        "planning_failure": failure_count,
        "success_rate": success_rate,
        "success_se": success_se,
        "success_rate_formatted": f"{success_rate:.4f} ± {success_se:.4f}",
        "failure_rate": failure_rate,
        "failure_se": failure_se,
        "failure_rate_formatted": f"{failure_rate:.4f} ± {failure_se:.4f}",
        "results": [asdict(r) for r in results]
    }
    
    # 실패 유형별 분류
    failure_types = {}
    for r in results:
        if not r.planning_success and 'error' not in r.planning_verdict.lower():
            verdict = r.planning_verdict
            failure_types[verdict] = failure_types.get(verdict, 0) + 1
    
    summary["failure_breakdown"] = failure_types
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Planning Error Analysis for Sokoban"
    )
    parser.add_argument(
        "report_file",
        type=str,
        help="Path to the report JSON file (e.g., report_pa.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: planning_analysis_<input>.json)"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200000,
        help="Maximum nodes for BFS planner (default: 200000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed analysis for each step"
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing: {args.report_file}")
    
    summary = analyze_report(
        args.report_file,
        max_nodes=args.max_nodes,
        verbose=args.verbose
    )
    
    if not summary:
        print("Analysis failed")
        return
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("PLANNING ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total steps analyzed: {summary['total_steps']}")
    print(f"Parse errors: {summary['parse_errors']}")
    print(f"Valid steps: {summary['valid_steps']}")
    print(f"Planning success: {summary['planning_success']}")
    print(f"Planning failure: {summary['planning_failure']}")
    print(f"Success rate: {summary['success_rate_formatted']}")
    print(f"Failure rate: {summary['failure_rate_formatted']}")
    
    if summary.get('failure_breakdown'):
        print("\nFailure breakdown:")
        for verdict, count in sorted(
            summary['failure_breakdown'].items(),
            key=lambda x: -x[1]
        ):
            print(f"  {verdict}: {count}")
    
    # 결과 저장
    if args.output:
        output_path = args.output
    else:
        input_name = Path(args.report_file).stem
        output_path = f"planning_analysis_{input_name}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

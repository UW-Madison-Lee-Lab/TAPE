import os
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel
from typing import Literal
from joblib import Parallel, delayed
from tqdm import tqdm
import math

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class ThoughtAnalysis(BaseModel):
    """LLM 분석 결과 스키마"""
    expected_next_action: Literal["U", "D", "L", "R", "Unknown"]


def extract_thought_and_action(content: str) -> tuple[str | None, str | None, str | None]:
    """
    content에서 Thought와 Action을 추출합니다.
    Returns: (thought_text, inferred_action, actual_action)
    """
    # Action 라인에서 실제 action 추출 (예: "Action: R" -> "R")
    action_match = re.search(r'Action:\s*([A-Za-z]+)', content)
    actual_action = action_match.group(1).strip() if action_match else None
    
    # Thought 부분 추출
    thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', content, re.DOTALL)
    thought_text = thought_match.group(1).strip() if thought_match else None
    
    # Rule-based로 Thought에서 추 inference되는 action 추출
    inferred_action = None
    if thought_text:
        inferred_action = infer_action_from_thought_rule_based(thought_text)
    
    return thought_text, inferred_action, actual_action


def extract_observation_from_user_message(content: str) -> tuple[str | None, int | None]:
    """
    user message에서 observation과 steps_remaining을 추출합니다.
    """
    observation = None
    steps_remaining = None
    
    # Observation 추출
    obs_match = re.search(r'Observation:\s*(.*?)(?=Steps remaining:|$)', content, re.DOTALL)
    if obs_match:
        observation = obs_match.group(1).strip()
    
    # Steps remaining 추출
    steps_match = re.search(r'Steps remaining:\s*(\d+)', content)
    if steps_match:
        steps_remaining = int(steps_match.group(1))
    
    return observation, steps_remaining


def infer_action_from_thought_rule_based(thought: str) -> str | None:
    """
    Rule-based: Thought 텍스트에서 의도된 action을 추론합니다.
    """
    thought_lower = thought.lower()
    
    # 방향 매핑 (다양한 표현 고려)
    direction_patterns = {
        'U': [
            r'\bup\b', r'\bmove up\b', r'\bgo up\b', r'\bpush up\b',
            r'\bupward\b', r'\bnorth\b', r'\b위\b', r'\b위로\b'
        ],
        'D': [
            r'\bdown\b', r'\bmove down\b', r'\bgo down\b', r'\bpush down\b',
            r'\bdownward\b', r'\bsouth\b', r'\b아래\b', r'\b아래로\b'
        ],
        'L': [
            r'\bleft\b', r'\bmove left\b', r'\bgo left\b', r'\bpush left\b',
            r'\bleftward\b', r'\bwest\b', r'\b왼쪽\b', r'\b왼쪽으로\b'
        ],
        'R': [
            r'\bright\b', r'\bmove right\b', r'\bgo right\b', r'\bpush right\b',
            r'\brightward\b', r'\beast\b', r'\b오른쪽\b', r'\b오른쪽으로\b'
        ]
    }
    
    # 마지막 문장에서 먼저 찾기 (의도가 마지막에 명시되는 경우가 많음)
    sentences = thought.split('.')
    last_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
    last_part = '.'.join(last_sentences).lower()
    
    # 마지막 부분에서 먼저 검색
    for action, patterns in direction_patterns.items():
        for pattern in patterns:
            if re.search(pattern, last_part):
                return action
    
    # 전체 텍스트에서 검색 (마지막 등장 기준)
    last_found = None
    last_pos = -1
    
    for action, patterns in direction_patterns.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, thought_lower))
            if matches:
                pos = matches[-1].end()
                if pos > last_pos:
                    last_pos = pos
                    last_found = action
    
    return last_found


def analyze_single_thought_with_llm(
    thought: str,
    model: str = "gpt-4.1-mini"
) -> str | None:
    """
    단일 Thought를 LLM으로 분석합니다. (병렬 처리용)
    """
    from real_agents.base import BaseAgent
    
    try:
        agent = BaseAgent(
            model=model,
            fine_tuned_model=None,
            temperature=0.0,
            is_print=False,
        )
        
        system_prompt = """You are an expert at analyzing reasoning text from a Sokoban puzzle game.
Your task is to determine what the IMMEDIATE NEXT action the player intends to take based on their thought process.

The available actions are:
- U: Move up (x, y) -> (x, y+1)
- D: Move down (x, y) -> (x, y-1)
- L: Move left (x, y) -> (x-1, y)
- R: Move right (x, y) -> (x+1, y)
- Unknown: Cannot determine the intended action from the thought

Analyze the thought carefully and identify what action they plan to execute NEXT (not later steps in a plan).
Focus on phrases like "I will move...", "I should go...", "Next I need to...", "My immediate action is..." etc.
If the thought does not clearly indicate a specific direction, return "Unknown"."""

        user_prompt = f"""Analyze this thought and determine the immediate next action:

Thought: {thought}

What is the immediate next action (U/D/L/R/Unknown) the player intends to take?"""

        response = agent.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "thought_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "expected_next_action": {
                                "type": "string",
                                "enum": ["U", "D", "L", "R", "Unknown"],
                                "description": "The immediate next action inferred from the thought, or Unknown if unclear"
                            }
                        },
                        "required": ["expected_next_action"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        result = json.loads(response[0])
        action = result.get("expected_next_action")
        # "Unknown"인 경우 None으로 변환하여 unknown 카운트에 포함
        return None if action == "Unknown" else action
        
    except Exception as e:
        print(f"LLM analysis error: {e}")
        return None


class ThoughtAnalyzer:
    """LLM을 사용한 Thought 분석기"""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        from real_agents.base import BaseAgent
        self.model = model
        self.agent = BaseAgent(
            model=model,
            fine_tuned_model=None,
            temperature=0.0,
            is_print=False,
        )
        
    def analyze(self, thought: str) -> str | None:
        """
        LLM을 사용하여 Thought에서 의도된 다음 action을 추론합니다.
        """
        system_prompt = """You are an expert at analyzing reasoning text from a Sokoban puzzle game.
Your task is to determine what the IMMEDIATE NEXT action the player intends to take based on their thought process.

The available actions are:
- U: Move up
- D: Move down  
- L: Move left
- R: Move right
- Unknown: Cannot determine the intended action from the thought

Analyze the thought carefully and identify what action they plan to execute NEXT (not later steps in a plan).
Focus on phrases like "I will move...", "I should go...", "Next I need to...", "My immediate action is..." etc.
If the thought does not clearly indicate a specific direction, return "Unknown"."""

        user_prompt = f"""Analyze this thought and determine the **immediate** next action:

Thought: {thought}

What is the immediate next action (U/D/L/R/Unknown) the player intends to take?"""

        try:
            response = self.agent.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "thought_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "expected_next_action": {
                                    "type": "string",
                                    "enum": ["U", "D", "L", "R", "Unknown"],
                                    "description": "The immediate next action inferred from the thought, or Unknown if unclear"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation of why this action was inferred"
                                }
                            },
                            "required": ["expected_next_action", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            result = json.loads(response[0])
            action = result.get("expected_next_action")
            # "Unknown"인 경우 None으로 변환하여 unknown 카운트에 포함
            return None if action == "Unknown" else action
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return None


def extract_all_thoughts_from_file(filepath: str) -> list[dict]:
    """
    파일에서 모든 Thought-Action 쌍을 추출합니다.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    thoughts = []
    
    if 'messages' in data:
        messages = data['messages']
    elif isinstance(data, list):
        messages = data
    else:
        return thoughts
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if not content or 'Thought:' not in content:
                continue
            
            thought_text, rule_inferred, actual = extract_thought_and_action(content)
            
            # 직전 user message에서 observation과 steps_remaining 추출
            observation = None
            steps_remaining = None
            if i > 0 and messages[i-1].get('role') == 'user':
                user_content = messages[i-1].get('content', '')
                observation, steps_remaining = extract_observation_from_user_message(user_content)
            
            thoughts.append({
                'step': i,
                'thought': thought_text,
                'rule_inferred': rule_inferred,
                'actual': actual,
                'content': content,
                'observation': observation,
                'steps_remaining': steps_remaining,
            })
    
    return thoughts


def analyze_file(
    filepath: str, 
    use_llm: bool = False, 
    llm_analyzer: ThoughtAnalyzer | None = None
) -> dict:
    """
    단일 파일을 분석하여 Thought-Action 불일치를 찾습니다.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {
        'total_steps': 0,
        'matched': 0,
        'mismatched': 0,
        'unknown': 0,
        'mismatches': [],
        'unknown_cases': [],
        'step_logs': [],  # 모든 step의 상세 로그
        'analysis_method': 'llm' if use_llm else 'rule-based'
    }
    
    # messages에서 assistant 응답 분석
    if 'messages' in data:
        messages = data['messages']
    elif isinstance(data, list):
        messages = data
    else:
        return results
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if not content or 'Thought:' not in content:
                continue
            
            results['total_steps'] += 1
            thought_text, rule_inferred, actual = extract_thought_and_action(content)
            
            # 직전 user message에서 observation과 steps_remaining 추출
            observation = None
            steps_remaining = None
            if i > 0 and messages[i-1].get('role') == 'user':
                user_content = messages[i-1].get('content', '')
                observation, steps_remaining = extract_observation_from_user_message(user_content)
            
            # LLM 또는 Rule-based 선택
            if use_llm and llm_analyzer and thought_text:
                inferred = llm_analyzer.analyze(thought_text)
            else:
                inferred = rule_inferred
            
            # actual_action이 None이고 inferred도 None(Unknown)인 경우 match로 처리
            if actual is None and inferred is None:
                is_match = True
            elif inferred is None:
                is_match = False
            else:
                is_match = (inferred == actual)
            
            # step log 저장
            step_log = {
                'step': i,
                'observation': observation,
                'steps_remaining': steps_remaining,
                'thought': thought_text[:500] if thought_text else None,
                'inferred_action': inferred if inferred else 'Unknown',
                'actual_action': actual,
                'is_match': is_match,
            }
            results['step_logs'].append(step_log)
            
            if is_match:
                results['matched'] += 1
                if inferred is None:
                    results['unknown'] += 1  # Unknown이지만 actual도 None이라 match
            elif inferred is None:
                # Unknown이고 actual은 있는 경우 -> mismatch
                results['unknown'] += 1
                results['mismatched'] += 1
                results['unknown_cases'].append({
                    'step': i,
                    'observation': observation,
                    'steps_remaining': steps_remaining,
                    'actual_action': actual,
                    'thought': thought_text[:500] if thought_text else None,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
                results['mismatches'].append({
                    'step': i,
                    'observation': observation,
                    'steps_remaining': steps_remaining,
                    'inferred_action': 'Unknown',
                    'actual_action': actual,
                    'thought': thought_text[:500] if thought_text else None,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
            else:
                results['mismatched'] += 1
                results['mismatches'].append({
                    'step': i,
                    'observation': observation,
                    'steps_remaining': steps_remaining,
                    'inferred_action': inferred,
                    'actual_action': actual,
                    'thought': thought_text[:500] if thought_text else None,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
    
    return results


def analyze_file_ours(filepath: str) -> dict:
    """
    'ours' 폴더용 파일 분석 함수.
    inferred_action = actual_action으로 처리하여 항상 match로 기록합니다.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {
        'total_steps': 0,
        'matched': 0,
        'mismatched': 0,
        'unknown': 0,
        'mismatches': [],
        'unknown_cases': [],
        'step_logs': [],
        'analysis_method': 'ours (inferred=actual)'
    }
    
    if 'messages' in data:
        messages = data['messages']
    elif isinstance(data, list):
        messages = data
    else:
        return results
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if not content or 'Thought:' not in content:
                continue
            
            results['total_steps'] += 1
            thought_text, _, actual = extract_thought_and_action(content)
            
            # 직전 user message에서 observation과 steps_remaining 추출
            observation = None
            steps_remaining = None
            if i > 0 and messages[i-1].get('role') == 'user':
                user_content = messages[i-1].get('content', '')
                observation, steps_remaining = extract_observation_from_user_message(user_content)
            
            # ours의 경우: inferred = actual (항상 match)
            inferred = actual
            is_match = True
            
            step_log = {
                'step': i,
                'observation': observation,
                'steps_remaining': steps_remaining,
                'thought': thought_text[:500] if thought_text else None,
                'inferred_action': inferred if inferred else 'Unknown',
                'actual_action': actual,
                'is_match': is_match,
            }
            results['step_logs'].append(step_log)
            results['matched'] += 1
    
    return results


def analyze_file_parallel(
    filepath: str,
    llm_results: dict[str, str] | None = None
) -> tuple[str, dict]:
    """
    병렬 처리용 파일 분석 함수.
    llm_results가 제공되면 LLM 결과를 사용, 아니면 rule-based.
    """
    filename = Path(filepath).name
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {
        'total_steps': 0,
        'matched': 0,
        'mismatched': 0,
        'unknown': 0,
        'mismatches': [],
        'unknown_cases': [],
        'step_logs': [],  # 모든 step의 상세 로그
        'analysis_method': 'llm' if llm_results else 'rule-based'
    }
    
    if 'messages' in data:
        messages = data['messages']
    elif isinstance(data, list):
        messages = data
    else:
        return filename, results
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if not content or 'Thought:' not in content:
                continue
            
            results['total_steps'] += 1
            thought_text, rule_inferred, actual = extract_thought_and_action(content)
            
            # 직전 user message에서 observation과 steps_remaining 추출
            observation = None
            steps_remaining = None
            if i > 0 and messages[i-1].get('role') == 'user':
                user_content = messages[i-1].get('content', '')
                observation, steps_remaining = extract_observation_from_user_message(user_content)
            
            # LLM 결과 사용 또는 rule-based
            key = f"{filename}_{i}"
            if llm_results and key in llm_results:
                inferred = llm_results[key]
            else:
                inferred = rule_inferred
            
            # actual_action이 None이고 inferred도 None(Unknown)인 경우 match로 처리
            if actual is None and inferred is None:
                is_match = True
            elif inferred is None:
                is_match = False
            else:
                is_match = (inferred == actual)
            
            # step log 저장
            step_log = {
                'step': i,
                'observation': observation,
                'steps_remaining': steps_remaining,
                'thought': thought_text[:500] if thought_text else None,
                'inferred_action': inferred if inferred else 'Unknown',
                'actual_action': actual,
                'is_match': is_match,
            }
            results['step_logs'].append(step_log)
            
            if is_match:
                results['matched'] += 1
                if inferred is None:
                    results['unknown'] += 1  # Unknown이지만 actual도 None이라 match
            elif inferred is None:
                # Unknown이고 actual은 있는 경우 -> mismatch
                results['unknown'] += 1
                results['mismatched'] += 1
                results['unknown_cases'].append({
                    'step': i,
                    'observation': observation,
                    'steps_remaining': steps_remaining,
                    'actual_action': actual,
                    'thought': thought_text[:500] if thought_text else None,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
                results['mismatches'].append({
                    'step': i,
                    'observation': observation,
                    'steps_remaining': steps_remaining,
                    'inferred_action': 'Unknown',
                    'actual_action': actual,
                    'thought': thought_text[:500] if thought_text else None,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
            else:
                results['mismatched'] += 1
                results['mismatches'].append({
                    'step': i,
                    'observation': observation,
                    'steps_remaining': steps_remaining,
                    'inferred_action': inferred,
                    'actual_action': actual,
                    'thought': thought_text[:500] if thought_text else None,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
    
    return filename, results


def calculate_stats(values: list[int]) -> tuple[float, float, int]:
    """
    이진 데이터(0/1)에 대한 평균과 Standard Error를 계산합니다.
    Returns: (mean, standard_error, n)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0
    
    mean = sum(values) / n
    
    # Standard Error for Binary Data (Bernoulli trial)
    # SE = sqrt( p * (1-p) / n )
    if n > 1:
        se = math.sqrt(mean * (1.0 - mean) / n)
    else:
        se = 0.0
    
    return mean, se, n


def print_summary(results: dict):
    """
    분석 결과 요약을 출력합니다.
    """
    total = max(results['total_steps'], 1)
    matched = results['total_matched']
    mismatched = results['total_mismatched']
    unknown = results['total_unknown']
    
    # 각 비율 계산
    match_rate = matched / total
    mismatch_rate = mismatched / total
    unknown_rate = unknown / total
    
    # Standard Error 계산 (이진 데이터)
    match_se = math.sqrt(match_rate * (1 - match_rate) / total) if total > 1 else 0.0
    mismatch_se = math.sqrt(mismatch_rate * (1 - mismatch_rate) / total) if total > 1 else 0.0
    unknown_se = math.sqrt(unknown_rate * (1 - unknown_rate) / total) if total > 1 else 0.0
    
    print("\n" + "=" * 60)
    print(f"📁 폴더: {results['folder']}")
    print(f"🔧 분석 방법: {results['analysis_method']}", end="")
    if results.get('llm_model'):
        print(f" ({results['llm_model']})")
    else:
        print()
    print("=" * 60)
    
    print(f"\n📊 전체 통계:")
    print(f"   - 분석한 파일 수: {results['total_files']}")
    print(f"   - 전체 스텝 수: {results['total_steps']}")
    print(f"   - ✅ 일치: {matched} ({match_rate*100:.1f}% ± {match_se*100:.1f}%)")
    print(f"   - ❌ 불일치 (Unknown 포함): {mismatched} ({mismatch_rate*100:.1f}% ± {mismatch_se*100:.1f}%)")
    print(f"   - ❓ 그 중 Unknown: {unknown} ({unknown_rate*100:.1f}% ± {unknown_se*100:.1f}%)")
    
    if results['all_mismatches']:
        print(f"\n🔍 불일치 상세 (최대 10개):")
        print("-" * 60)
        for i, mismatch in enumerate(results['all_mismatches'][:10]):
            print(f"\n[{i+1}] 파일: {mismatch['file']}, Step: {mismatch['step']}")
            print(f"    추론된 Action: {mismatch['inferred_action']} → 실제 Action: {mismatch['actual_action']}")
            if mismatch.get('thought'):
                thought_preview = mismatch['thought'][:200].replace('\n', ' ')
                print(f"    Thought: {thought_preview}...")
        
        if len(results['all_mismatches']) > 10:
            print(f"\n    ... 외 {len(results['all_mismatches']) - 10}개 더 있음")


def print_file_breakdown(results: dict):
    """
    파일별 상세 결과를 출력합니다.
    """
    print("\n" + "=" * 60)
    print("📄 파일별 상세 결과:")
    print("=" * 60)
    
    for filename, file_results in sorted(results['files'].items()):
        total = file_results['total_steps']
        if total == 0:
            continue
            
        matched = file_results['matched']
        mismatched = file_results['mismatched']
        unknown = file_results['unknown']
        
        status = "✅" if mismatched == 0 else "⚠️" if mismatched < total * 0.3 else "❌"
        print(f"\n{status} {filename}")
        print(f"   Steps: {total} | 일치: {matched} | 불일치: {mismatched} | 추론불가: {unknown}")
        
        if file_results['mismatches']:
            for m in file_results['mismatches'][:3]:
                print(f"      - Step {m['step']}: {m['inferred_action']} → {m['actual_action']}")


def recalculate_stats_from_file(filepath: str) -> dict:
    """
    기존 결과 파일을 읽어서 stats만 다시 계산합니다.
    Unknown도 mismatch에 포함하여 재계산합니다.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # unknown을 mismatch에 포함하여 재계산
    total = max(data['total_steps'], 1)
    matched = data['total_matched']
    unknown = data['total_unknown']
    
    # mismatch = 기존 mismatch + unknown (이미 포함되어 있지 않은 경우)
    # 새 로직: mismatched는 이미 unknown을 포함
    mismatched = data['total_mismatched']
    
    # 비율 계산
    match_rate = matched / total
    mismatch_rate = mismatched / total
    unknown_rate = unknown / total
    
    # Standard Error 계산
    match_se = math.sqrt(match_rate * (1 - match_rate) / total) if total > 1 else 0.0
    mismatch_se = math.sqrt(mismatch_rate * (1 - mismatch_rate) / total) if total > 1 else 0.0
    unknown_se = math.sqrt(unknown_rate * (1 - unknown_rate) / total) if total > 1 else 0.0
    
    # 기존 데이터 업데이트
    data['match_rate'] = match_rate
    data['match_se'] = match_se
    data['match_rate_formatted'] = f"{match_rate:.3f} ± {match_se:.3f}"
    data['mismatch_rate'] = mismatch_rate
    data['mismatch_se'] = mismatch_se
    data['mismatch_rate_formatted'] = f"{mismatch_rate:.3f} ± {mismatch_se:.3f}"
    data['unknown_rate'] = unknown_rate
    data['unknown_se'] = unknown_se
    data['unknown_rate_formatted'] = f"{unknown_rate:.3f} ± {unknown_se:.3f}"
    
    # files_summary도 업데이트
    if 'files_summary' in data:
        for name, r in data['files_summary'].items():
            file_total = max(r.get('total', 1), 1)
            file_unknown = r.get('unknown', 0)
            file_mismatched = r.get('mismatched', 0)
            # unknown을 mismatch에 포함 (이미 포함되어 있으면 그대로)
            r['match_rate'] = r.get('matched', 0) / file_total
            r['mismatch_rate'] = file_mismatched / file_total
    
    return data


def analyze_folder(
    folder_path: str, 
    use_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
    n_jobs: int = -1,
    verbose: bool = True
) -> dict:
    """
    폴더 내의 모든 JSON 파일을 분석합니다.
    use_llm=True이고 n_jobs != 1이면 joblib으로 병렬 처리합니다.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: 폴더가 존재하지 않습니다: {folder_path}")
        return {}

    # "ours"가 경로에 포함되어 있으면 inferred_action = actual_action으로 처리
    is_ours = "ours" in folder_path.lower()
    
    if is_ours:
        analysis_method = 'ours (inferred=actual)'
    elif use_llm:
        analysis_method = 'llm'
    else:
        analysis_method = 'rule-based'
    
    all_results = {
        'folder': folder_path,
        'analysis_method': analysis_method,
        'llm_model': llm_model if use_llm and not is_ours else None,
        'total_files': 0,
        'total_steps': 0,
        'total_matched': 0,
        'total_mismatched': 0,
        'total_unknown': 0,
        'files': {},
        'all_mismatches': []
    }
    
    json_files = sorted(list(folder.glob('*.json')))
    
    if not json_files:
        print(f"Warning: JSON 파일을 찾을 수 없습니다: {folder_path}")
        return all_results
    
    if is_ours:
        # "ours" 폴더의 경우: inferred_action = actual_action으로 처리 (항상 match)
        if verbose:
            print(f"📋 'ours' 폴더 감지: inferred_action = actual_action으로 처리")
        
        for idx, json_file in enumerate(json_files):
            if verbose:
                print(f"\r📄 분석 중: {idx + 1}/{len(json_files)} - {json_file.name}", end="")
            
            try:
                file_results = analyze_file_ours(str(json_file))
                all_results['total_files'] += 1
                all_results['total_steps'] += file_results['total_steps']
                all_results['total_matched'] += file_results['matched']
                all_results['total_mismatched'] += file_results['mismatched']
                all_results['total_unknown'] += file_results['unknown']
                all_results['files'][json_file.name] = file_results
                
                for mismatch in file_results['mismatches']:
                    mismatch['file'] = json_file.name
                    all_results['all_mismatches'].append(mismatch)
                    
            except Exception as e:
                if verbose:
                    print(f"\nError processing {json_file}: {e}")
        
        if verbose:
            print()
    
    elif use_llm:
        # LLM 병렬 처리
        print(f"🤖 LLM 분석 시작 (model: {llm_model}, jobs: {n_jobs})")
        
        # Step 1: 모든 파일에서 thoughts 추출
        all_thoughts = []
        for json_file in json_files:
            thoughts = extract_all_thoughts_from_file(str(json_file))
            for t in thoughts:
                t['file'] = json_file.name
                t['key'] = f"{json_file.name}_{t['step']}"
            all_thoughts.extend(thoughts)
        
        print(f"📝 총 {len(all_thoughts)}개의 Thought 발견")
        
        # Step 2: joblib으로 병렬 LLM 분석
        def analyze_thought_wrapper(thought_data: dict) -> tuple[str, str | None]:
            
            if thought_data['thought']:
                result = analyze_single_thought_with_llm(
                    thought_data['thought'],
                    model=llm_model
                )
                return thought_data['key'], result
            return thought_data['key'], None
        
        if verbose:
            print(f"🔄 LLM 병렬 분석 중...")
        
        llm_results_list = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(analyze_thought_wrapper)(t) 
            for t in tqdm(all_thoughts, desc="LLM 분석", disable=not verbose)
        )
        
        llm_results = dict(llm_results_list)
        
        # Step 3: 결과 집계
        file_results_list = Parallel(n_jobs=n_jobs)(
            delayed(analyze_file_parallel)(str(json_file), llm_results)
            for json_file in tqdm(json_files, desc="결과 집계", disable=not verbose)
        )
        
        for filename, file_results in file_results_list:
            all_results['total_files'] += 1
            all_results['total_steps'] += file_results['total_steps']
            all_results['total_matched'] += file_results['matched']
            all_results['total_mismatched'] += file_results['mismatched']
            all_results['total_unknown'] += file_results['unknown']
            all_results['files'][filename] = file_results
            
            for mismatch in file_results['mismatches']:
                mismatch['file'] = filename
                all_results['all_mismatches'].append(mismatch)
    
    else:
        # Rule-based (기존 로직)
        for idx, json_file in enumerate(json_files):
            if verbose:
                print(f"\r📄 분석 중: {idx + 1}/{len(json_files)} - {json_file.name}", end="")
            
            try:
                file_results = analyze_file(str(json_file), use_llm=False)
                all_results['total_files'] += 1
                all_results['total_steps'] += file_results['total_steps']
                all_results['total_matched'] += file_results['matched']
                all_results['total_mismatched'] += file_results['mismatched']
                all_results['total_unknown'] += file_results['unknown']
                all_results['files'][json_file.name] = file_results
                
                for mismatch in file_results['mismatches']:
                    mismatch['file'] = json_file.name
                    all_results['all_mismatches'].append(mismatch)
                    
            except Exception as e:
                if verbose:
                    print(f"\nError processing {json_file}: {e}")
        
        if verbose:
            print()
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Thought-Action 불일치 분석 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Rule-based 분석 (기본)
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react
  
  # LLM 기반 분석 (병렬 처리)
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react --use-llm
  
  # 병렬 작업 수 지정
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react --use-llm --n-jobs 8
  
  # 특정 LLM 모델 사용
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react --use-llm --llm-model gpt-4o
  
  # 상세 결과 출력
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react --detailed
  
  # 결과 JSON으로 저장
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react --output report.json
  
  # 기존 파일이 있으면 stats만 재계산
  python sampling_error_analysis.py results/sokoban_2/gpt-4.1-mini/react --output report.json --update-only
        """
    )
    parser.add_argument('folder', help='분석할 폴더 경로')
    parser.add_argument('--use-llm', action='store_true', 
                        help='LLM을 사용하여 Thought 분석 (기본: rule-based)')
    parser.add_argument('--llm-model', default='gpt-4.1-mini',
                        help='LLM 분석에 사용할 모델 (기본: gpt-4.1-mini)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='병렬 작업 수 (-1: 모든 CPU 사용, 기본: -1)')
    parser.add_argument('--detailed', '-d', action='store_true', 
                        help='파일별 상세 결과 출력')
    parser.add_argument('--output', '-o', help='결과를 JSON 파일로 저장')
    parser.add_argument('--show-unknown', action='store_true', 
                        help='추론 불가 케이스도 상세 출력')
    parser.add_argument('--update-only', '-u', action='store_true',
                        help='기존 출력 파일이 있으면 stats만 재계산하여 업데이트')
    
    args = parser.parse_args()
    
    # --update-only 모드: 기존 파일이 있으면 stats만 재계산
    if args.update_only and args.output and os.path.exists(args.output):
        print(f"📄 기존 파일 발견: {args.output}")
        print(f"🔄 Stats 재계산 중...")
        
        try:
            updated_data = recalculate_stats_from_file(args.output)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Stats 업데이트 완료!")
            print(f"   - 전체 스텝: {updated_data['total_steps']}")
            print(f"   - ✅ 일치: {updated_data['match_rate_formatted']}")
            print(f"   - ❌ 불일치: {updated_data['mismatch_rate_formatted']}")
            print(f"   - ❓ 추론 불가: {updated_data['unknown_rate_formatted']}")
            print(f"\n💾 저장됨: {args.output}")
            return
            
        except Exception as e:
            print(f"⚠️ 기존 파일 업데이트 실패: {e}")
            print(f"🔄 전체 분석을 다시 실행합니다...")
    
    method = "LLM" if args.use_llm else "Rule-based"
    print(f"🔍 분석 시작: {args.folder}")
    print(f"🔧 분석 방법: {method}")
    if args.use_llm:
        print(f"⚡ 병렬 작업 수: {args.n_jobs}")
    
    results = analyze_folder(
        args.folder,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        n_jobs=args.n_jobs
    )
    
    if not results or results['total_files'] == 0:
        print("분석할 파일이 없습니다.")
        return
    
    print_summary(results)
    
    if args.detailed:
        print_file_breakdown(results)
    
    if args.show_unknown and results.get('total_unknown', 0) > 0:
        print("\n" + "=" * 60)
        print("❓ 추론 불가 케이스 (최대 5개):")
        print("=" * 60)
        count = 0
        for filename, file_results in results['files'].items():
            for case in file_results.get('unknown_cases', []):
                if count >= 5:
                    break
                print(f"\n파일: {filename}, Step: {case['step']}")
                print(f"실제 Action: {case['actual_action']}")
                if case.get('observation'):
                    print(f"Observation: {case['observation'][:200]}...")
                if case.get('steps_remaining') is not None:
                    print(f"Steps Remaining: {case['steps_remaining']}")
                if case.get('thought'):
                    thought_preview = case['thought'][:300].replace('\n', ' ')
                    print(f"Thought: {thought_preview}...")
                count += 1
    
    if args.output:
        total = max(results['total_steps'], 1)
        match_rate = results['total_matched'] / total
        mismatch_rate = results['total_mismatched'] / total
        unknown_rate = results['total_unknown'] / total
        
        # Standard Error 계산
        match_se = math.sqrt(match_rate * (1 - match_rate) / total) if total > 1 else 0.0
        mismatch_se = math.sqrt(mismatch_rate * (1 - mismatch_rate) / total) if total > 1 else 0.0
        unknown_se = math.sqrt(unknown_rate * (1 - unknown_rate) / total) if total > 1 else 0.0
        
        # 모든 step_logs 수집
        all_step_logs = []
        for filename, file_results in results['files'].items():
            for log in file_results.get('step_logs', []):
                log['file'] = filename
                all_step_logs.append(log)
        
        save_results = {
            'folder': results['folder'],
            'analysis_method': results['analysis_method'],
            'llm_model': results.get('llm_model'),
            'total_files': results['total_files'],
            'total_steps': results['total_steps'],
            'total_matched': results['total_matched'],
            'total_mismatched': results['total_mismatched'],
            'total_unknown': results['total_unknown'],
            # 비율 및 Standard Error 추가
            'match_rate': match_rate,
            'match_se': match_se,
            'match_rate_formatted': f"{match_rate:.3f} ± {match_se:.3f}",
            'mismatch_rate': mismatch_rate,
            'mismatch_se': mismatch_se,
            'mismatch_rate_formatted': f"{mismatch_rate:.3f} ± {mismatch_se:.3f}",
            'unknown_rate': unknown_rate,
            'unknown_se': unknown_se,
            'unknown_rate_formatted': f"{unknown_rate:.3f} ± {unknown_se:.3f}",
            'files_summary': {
                name: {
                    'total': r['total_steps'],
                    'matched': r['matched'],
                    'mismatched': r['mismatched'],
                    'unknown': r['unknown'],
                    'match_rate': r['matched'] / max(r['total_steps'], 1),
                    'mismatch_rate': r['mismatched'] / max(r['total_steps'], 1),
                }
                for name, r in results['files'].items()
            },
            # 모든 step의 상세 로그 (observation, steps_remaining, inferred, actual 포함)
            'all_step_logs': all_step_logs,
            'all_mismatches': [
                {
                    'file': m['file'],
                    'step': m['step'],
                    'observation': m.get('observation', '')[:300] if m.get('observation') else None,
                    'steps_remaining': m.get('steps_remaining'),
                    'inferred': m['inferred_action'],
                    'actual': m['actual_action'],
                    'thought': m.get('thought', '')[:300]
                }
                for m in results['all_mismatches']
            ]
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장됨: {args.output}")


if __name__ == '__main__':
    main()

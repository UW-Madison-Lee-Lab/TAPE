import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.alfworld import StateAlfworld

def test_state_equality():
    """StateAlfworld 객체 비교 테스트"""
    
    # 테스트 1: 완전히 동일한 객체
    state1 = StateAlfworld(
        inventory=(("apple", "heated"), ("egg", "raw")),
        your_location="middle of the room",
        subgoal_progress="Your subgoal progress is before cooling tomato."
    )
    state2 = StateAlfworld(
        inventory=(("apple", "heated"), ("egg", "raw")),
        your_location="middle of the room",
        subgoal_progress="Your subgoal progress is before cooling tomato."
    )
    
    print("=== 테스트 1: 동일한 객체 ===")
    print(f"state1: {repr(state1)}")
    print(f"state2: {repr(state2)}")
    print(f"state1 == state2: {state1 == state2}")
    print(f"hash(state1) == hash(state2): {hash(state1) == hash(state2)}")
    print(f"hash(state1): {hash(state1)}")
    print(f"hash(state2): {hash(state2)}")
    
    # 딕셔너리 테스트
    d = {state1: 10}
    print(f"d[state2] = {d.get(state2)}")  # 10이 나와야 함
    print()
    
    # 테스트 2: inventory 순서가 다른 경우
    state3 = StateAlfworld(
        inventory=(("egg", "raw"), ("apple", "heated")),  # 순서 다름!
        your_location="middle of the room",
        subgoal_progress="Your subgoal progress is before cooling tomato."
    )
    
    print("=== 테스트 2: inventory 순서 다름 ===")
    print(f"state1.inventory: {state1.inventory}")
    print(f"state3.inventory: {state3.inventory}")
    print(f"state1 == state3: {state1 == state3}")  # False
    print(f"d.get(state3): {d.get(state3)}")  # None
    print()
    
    # 테스트 3: 공백 차이
    state4 = StateAlfworld(
        inventory=(("apple", "heated"), ("egg", "raw")),
        your_location="middle of the room ",  # 끝에 공백!
        subgoal_progress="Your subgoal progress is before cooling tomato."
    )
    
    print("=== 테스트 3: 공백 차이 ===")
    print(f"state1.your_location: {repr(state1.your_location)}")
    print(f"state4.your_location: {repr(state4.your_location)}")
    print(f"state1 == state4: {state1 == state4}")  # False
    print(f"d.get(state4): {d.get(state4)}")  # None
    print()
    
    # 테스트 4: 빈 inventory
    state5 = StateAlfworld(
        inventory=(),
        your_location="middle of the room",
        subgoal_progress="Your subgoal progress is before cooling tomato."
    )
    state6 = StateAlfworld(
        inventory=tuple(),
        your_location="middle of the room",
        subgoal_progress="Your subgoal progress is before cooling tomato."
    )
    
    print("=== 테스트 4: 빈 inventory ===")
    print(f"state5.inventory: {repr(state5.inventory)}")
    print(f"state6.inventory: {repr(state6.inventory)}")
    print(f"state5 == state6: {state5 == state6}")  # True
    print()
    
    # 테스트 5: set/dict에서의 동작
    print("=== 테스트 5: set/dict 동작 ===")
    V = {state1, state2}  # 같은 객체이므로 1개만
    print(f"len({{state1, state2}}): {len(V)}")  # 1
    
    out_degree = {state1: 0}
    result = out_degree.get(state2)
    print(f"out_degree[state1] = 0, out_degree.get(state2) = {result}")  # 0
    
    result3 = out_degree.get(state3)
    print(f"out_degree.get(state3) (순서 다름) = {result3}")  # None


def debug_actual_states(out_degree, test_state):
    """실제 out_degree와 테스트할 state를 비교"""
    print("=== 실제 데이터 디버깅 ===")
    print(f"test_state: {repr(test_state)}")
    print(f"test_state hash: {hash(test_state)}")
    print()
    
    for i, (k, v) in enumerate(out_degree.items()):
        print(f"--- Key {i} ---")
        print(f"  type: {type(k)}")
        print(f"  repr: {repr(k)}")
        print(f"  hash: {hash(k)}")
        print(f"  == test_state: {k == test_state}")
        
        # 필드별 비교
        print(f"  inventory 같음: {k.inventory == test_state.inventory}")
        print(f"  your_location 같음: {k.your_location == test_state.your_location}")
        print(f"  subgoal_progress 같음: {k.subgoal_progress == test_state.subgoal_progress}")
        
        # 문자열 상세 비교
        if k.your_location != test_state.your_location:
            print(f"    k.your_location: {repr(k.your_location)}")
            print(f"    test.your_location: {repr(test_state.your_location)}")
        if k.subgoal_progress != test_state.subgoal_progress:
            print(f"    k.subgoal_progress: {repr(k.subgoal_progress)}")
            print(f"    test.subgoal_progress: {repr(test_state.subgoal_progress)}")
        print()


if __name__ == "__main__":
    test_state_equality()

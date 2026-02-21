from typing import Optional, Dict, Any, List, Tuple
import json
import os
import yaml
import re
from pathlib import Path

from alfworld.agents.environment import get_environment
from dataclasses import dataclass

class StateAlfworld:
    __slots__ = ('inventory', 'your_location', 'observation', "subgoal_progress")

    def __init__(self, inventory, your_location: str, observation: str, subgoal_progress: str):
        """생성 시 자동으로 inventory 정렬 및 문자열 정규화"""
        # inventory 정규화: 정렬하고 tuple로 변환
        normalized_inventory = tuple(sorted(inventory))
        normalized_location = your_location.strip()
        observation = observation.strip()
        subgoal_progress = subgoal_progress.strip()
        
        # frozen처럼 동작하게 object.__setattr__ 사용
        object.__setattr__(self, 'inventory', normalized_inventory)
        object.__setattr__(self, 'your_location', normalized_location)
        object.__setattr__(self, 'observation', observation)
        object.__setattr__(self, 'subgoal_progress', subgoal_progress)

    def __setattr__(self, name, value):
        raise AttributeError("StateAlfworld is immutable")
    
    def __delattr__(self, name):
        raise AttributeError("StateAlfworld is immutable")
    
    def __eq__(self, other):
        if not isinstance(other, StateAlfworld):
            return False
        return (
            self.inventory == other.inventory
            and self.your_location == other.your_location
            and self.observation == other.observation
            and self.subgoal_progress == other.subgoal_progress
        )
    
    def __hash__(self):
        return hash((self.inventory, self.your_location, self.observation, self.subgoal_progress))
    
    def __repr__(self):
        return (
            f"StateAlfworld(inventory={self.inventory!r}, "
            f"your_location={self.your_location!r}, "
            f"observation={self.observation!r}, "
            f"subgoal_progress={self.subgoal_progress!r})"
        )

    def to_observation(self) -> Dict[str, Any]:
        inventory = [{"object": obj, "status": status} for obj, status in self.inventory]
        return {
            "inventory": inventory,
            "your location": self.your_location,
            "observation": self.observation,
            "subgoal_progress": self.subgoal_progress,
        }

def _decode_pddl_name(name: str) -> str:
    """
    'Desk_bar__minus_00_dot_57_bar__plus_00_dot_00_bar__minus_01_dot_35'
    -> 'Desk'
    """
    return name.split("_bar__", 1)[0]

def _natural_prod(item):
    """
    Natural sort helper.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', item)]

def load_recep_to_objects_from_pddl(gamefile_path: str) -> Dict[str, List[str]]:
    print(gamefile_path)
    """
    initial_state.pddl을 파싱하여, 각 Receptacle 인스턴스별(예: drawer 1, drawer 2)로 
    포함된 Object 목록(예: pen 1, key 1)을 반환합니다.
    
    사용자의 요청에 따라:
    1. Type 이름은 소문자로 변환하여 사용 (예: SinkBasin -> sinkbasin)
    2. Instance ID 정렬은 자연 정렬(Natural Sort) 사용
    """
    pddl_dir = os.path.dirname(gamefile_path)
    init_pddl_path = os.path.join(pddl_dir, "initial_state.pddl")
    recep_map: Dict[str, List[str]] = {}

    if not os.path.exists(init_pddl_path):
        return recep_map

    with open(init_pddl_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. 모든 Object 정의 수집 (ID 부여를 위해)
    objects_match = re.search(r"\(:objects\s+(.*?)\)", text, re.S)
    all_instances_by_type: Dict[str, List[str]] = {}
    
    if objects_match:
        obj_content = objects_match.group(1)
        tokens = obj_content.split()
        for t in tokens:
            if t.startswith("-") or t in ["object", ")", "("]: continue
            if "_bar__" in t:
                t_name = _decode_pddl_name(t)
                if t_name == "Sink":
                    t_name = "SinkBasin"
                all_instances_by_type.setdefault(t_name, []).append(t)

    # 2. Receptacle 초기화
    # 실제 Receptacle로 쓰일 수 있는 타입들
    container_types = {
        "ArmChair", "Bed", "Cabinet", "CoffeeTable", "CounterTop", "Desk", 
        "DiningTable", "Drawer", "Dresser", "Fridge", "GarbageCan", 
        "HandTowelHolder", "LaundryHamper", "Microwave", "Ottoman", "PaintingHanger",
        "Pan", "Pot", "Safe", "Shelf", "SideTable", "SinkBasin", "Sofa", 
        "StoveBurner", "Toilet", "ToiletPaperHanger", "TowelHolder", "TVStand", "Bathtub", "BathtubBasin",
        "Cart", "Sink"
    }

    # 3. 정렬 후 번호 할당 (Receptacle/Object 모두 역순 번호)
    # 일반적인 String Sort는 "Item_1", "Item_10", "Item_2" 순서가 되어 index가 꼬일 수 있음.
    pddl_to_readable = {}
    for t_name, insts in all_instances_by_type.items():
        insts.sort(key=_natural_prod, reverse=True)

        # 타입 이름을 소문자로 (예: SinkBasin -> sinkbasin)
        base_name = t_name.lower()

        for idx, inst in enumerate(insts):
            readable_name = f"{base_name} {idx+1}"
            pddl_to_readable[inst] = readable_name
    
    for pddl_inst, readable in pddl_to_readable.items():
        t_name = _decode_pddl_name(pddl_inst)
        if t_name in container_types:
            recep_map[readable] = []

    # 4. 관계(inReceptacle, on) 파싱하여 내용물 추가
    m = re.search(r"\(:init(.*)\)\s*\Z", text, re.S)
    if m:
        init_block = m.group(1)
        pattern = re.compile(r"\(\s*(inReceptacle|on)\s+([^\s]+)\s+([^\s]+)\s*\)")
        for pred, obj_inst, recep_inst in pattern.findall(init_block):
            if obj_inst in pddl_to_readable and recep_inst in pddl_to_readable:
                r_name = pddl_to_readable[recep_inst]
                o_name = pddl_to_readable[obj_inst]
            
                if r_name in recep_map:
                    recep_map[r_name].append(o_name)
    
    # 보기 좋게 자연 정렬
    for r in recep_map:
        recep_map[r].sort(key=_natural_prod)
    
    return dict(sorted(recep_map.items(), key=lambda item: _natural_prod(item[0])))

def get_gold_plan_length(gamefile_path: str) -> int:
    """
    gamefile_path와 같은 디렉토리의 traj_data.json을 로드하여,
    가장 마지막 low_idx + 1 또는 plan 길이를 이용하여 expert plan 길이를 추정합니다.
    (traj_data.json의 images 리스트에서 low_idx가 증가하는 형태를 보임)
    """
    data_dir = os.path.dirname(gamefile_path)
    traj_path = os.path.join(data_dir, "traj_data.json")

    if not os.path.exists(traj_path):
        return -1
    
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 0. High Level PDDL Plan 확인 (가장 정확한 ALFWorld Step 수)
        # plan -> high_pddl 리스트의 길이가 곧 High Level Action의 수
        if "plan" in data and "high_pddl" in data["plan"]:
            return len(data["plan"]["high_pddl"])
    except Exception:
        return [[0]]

class AlfworldEnv:
    def __init__(
        self,
        config: str = str(Path(__file__).resolve().parents[2] / "data" / "mini_config.yaml"),
        split: str = "eval_out_of_distribution",
        batch_size: int = 1,
    ):
        with open(config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        env_class = get_environment(self.config["env"]["type"])
        self.split = split
        self.env = env_class(self.config, train_eval=self.split).init_env(batch_size=batch_size)

    def reset(self):
        obs_list, info = self.env.reset()
        recep_to_objs: Dict[str, List[str]] = {}
        transition_info_txt = ""

        gamefile_path: Optional[str] = None
        if isinstance(info, dict):
            gamefiles = info.get("extra.gamefile")
            if isinstance(gamefiles, (list, tuple)) and gamefiles:
                gamefile_path = gamefiles[0]

        if gamefile_path:
            recep_to_objs = load_recep_to_objects_from_pddl(gamefile_path)
            lines = []
            for k, vs in (recep_to_objs or {}).items():
                if not vs:
                    lines.append(f"At {k.lower()}, there is nothing.")
                    continue
                noun = vs[0].lower() if len(vs) == 1 else ", ".join([v.lower() for v in vs])
                verb = "is" if len(vs) == 1 else "are"
                lines.append(f"At {k.lower()}, there {verb} {noun}.")
            transition_info_txt = "\n".join(lines)

        return obs_list, info, recep_to_objs, transition_info_txt

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

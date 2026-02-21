from real_agents.ours_graph_ilp import OursGraphILPAgent


class OursGraphILPAgentWOSolverReplanning(OursGraphILPAgent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("use_solver", False)
        kwargs.setdefault("use_replanning", False)
        kwargs.setdefault("use_strong_conditioning", False)
        super().__init__(*args, **kwargs)

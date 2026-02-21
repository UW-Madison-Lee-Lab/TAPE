from real_agents.ours_graph_ilp import OursGraphILPAgent


class OursGraphILPAgentWOReplanningStrongConditioning(OursGraphILPAgent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("use_solver", True)
        kwargs.setdefault("use_replanning", False)
        kwargs.setdefault("use_strong_conditioning", True)
        super().__init__(*args, **kwargs)

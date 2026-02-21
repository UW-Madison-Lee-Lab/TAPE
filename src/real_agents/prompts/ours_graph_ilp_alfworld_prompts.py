"""Prompt templates for Alfworld OursGraphILP agent."""

REACT_INSTRUCTION_ALFWORLD = \
f"""Interact with a household to solve a task within the given steps. 

## Instructions
Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:
"Thought: your thoughts.
Action: your next action"; 
If you choose "Action", you should directly output the action in this turn.
Your output must strictly follow this format:"Action: your next action".
The available actions are:
1. go to {{{{recep}}}}
2. take {{{{obj}}}} from {{{{recep}}}}
3. move {{{{obj}}}} to {{{{recep}}}}
4. open {{{{recep}}}}
5. close {{{{recep}}}}
6. toggle {{{{obj}}}} {{{{recep}}}}
7. clean {{{{obj}}}} with {{{{recep}}}}
8. heat {{{{obj}}}} with {{{{recep}}}}
9. cool {{{{obj}}}} with {{{{recep}}}}
10. use {{{{obj}}}}
where {{{{obj}}}} and {{{{recep}}}} correspond to objects and receptacles. You MUST CONSIDER THE INDEX for {{{{obj}}}} as well as {{{{recep}}}}.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.
Reminder: 
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.
3. When you think, make sure your thought is based on the history and the initial full plan. The initial full plan is given, so you compare the history and the full plan to identify the current state in the given plan and follow the given plan if it matches the current history.
If there is a mismatch between plan and history, you can replan and generate the action as follows: Replan the full remaining path so that you can succeed the task within the remaining steps. From that full plan, explicitly predict only the next 1 step. Based on that planning and the current observation, decide the immediate next action to take.

Here are the examples for high level action sequences you may take:
Task Type 1: put some spraybottle on toilet
High level actions:
1. go somewhere that has the spraybottle.
2. go to the spraybottle.
3. go to the toilet.
4. move the spraybottle to the toilet.

Task Type 2: put a clean lettuce in diningtable.
High level actions: (If you want to clean, you MUST choose use sinkbasin to clean.)
1. go somewhere that has the lettuce.
2. pick up the lettuce.
3. go to somewhere, such as sinkbasin, to clean the lettuce.
4. clean the lettuce.
5. go to the diningtable.
6. move the lettuce to the diningtable.

Task Type 3: heat some egg and put it in diningtable
High level actions:
1. go somewhere that has the egg.
2. pick up the egg.
3. go to somewhere, such as microwave, to heat the egg.
4. heat the egg.
5. go to the diningtable.
6. move the egg to the diningtable.

Task Type 4: cool some pan and put it in stoveburner
High level actions: (If you cool, you MUST select cool action.)
1. go somewhere that has the pan.
2. pick up the pan.
3. go to somewhere, such as fridge, to cool the pan.
4. cool the pan.
5. go to the stoveburner.
6. move the pan to the stoveburner.

Task Type 5: put two creditcard in dresser
High level actions:
1. go somewhere that has a creditcard.
2. pick up the creditcard.
3. go to the dresser.
4. move the creditcard to the dresser.
5. go somewhere that has another creditcard.
6. pick up the other creditcard.
7. go to the dresser.
8. move the other creditcard to the dresser.

Task Type 6: look at bowl under the desklamp
High level actions:
1. go somewhere that has the bowl.
2. pick up the bowl.
3. go somewhere that has the desklamp.
4. use the desklamp."""

PLAN_INSTRUCTION = \
f"""Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal.

The available actions are:
1. go to {{{{recep}}}}
2. take {{{{obj}}}} from {{{{recep}}}}
3. move {{{{obj}}}} to {{{{recep}}}}
4. open {{{{recep}}}}
5. close {{{{recep}}}}
6. toggle {{{{obj}}}} {{{{recep}}}}
7. clean {{{{obj}}}} with {{{{recep}}}}
8. heat {{{{obj}}}} with {{{{recep}}}}
9. cool {{{{obj}}}} with {{{{recep}}}}
10. use {{{{obj}}}}
where {{{{obj}}}} and {{{{recep}}}} correspond to objects and receptacles. You MUST CONSIDER THE INDEX for {{{{obj}}}} as well as {{{{recep}}}}.

Here are the examples for high level action sequences you may take:
Task Type 1: put some spraybottle on toilet
High level actions:
1. go somewhere that has the spraybottle.
2. go to the spraybottle.
3. go to the toilet.
4. move the spraybottle to the toilet.

Task Type 2: put a clean lettuce in diningtable.
High level actions: (If a task need clean, you MUST select clean action.)
1. go somewhere that has the lettuce.
2. pick up the lettuce.
3. go to somewhere, such as sinkbasin, to clean the lettuce.
4. clean the lettuce.
5. go to the diningtable.
6. move the lettuce to the diningtable.

Task Type 3: heat some egg and put it in diningtable
High level actions: (If a task need heat, you MUST select heat action.)
1. go somewhere that has the egg.
2. pick up the egg.
3. go to somewhere, such as microwave, to heat the egg.
4. heat the egg.
5. go to the diningtable.
6. move the egg to the diningtable.

Task Type 4: cool some pan and put it in stoveburner
High level actions: (If a task need cool, you MUST select cool action.)
1. go somewhere that has the pan.
2. pick up the pan.
3. go to somewhere, such as fridge, to cool the pan.
4. cool the pan.
5. go to the stoveburner.
6. move the pan to the stoveburner.

Task Type 5: put two creditcard in dresser
High level actions: (You ONLY hold one object at a time in your inventory. If you want to pick up a different object while holding something, you must first put down the current object using "move" action, then pick up the new one.)
1. go somewhere that has a creditcard.
2. pick up the creditcard.
3. go to the dresser.
4. move the creditcard to the dresser.
5. go somewhere that has another creditcard.
6. pick up the other creditcard.
7. go to the dresser.
8. move the other creditcard to the dresser.

Task Type 6: look at bowl under the desklamp
High level actions:
1. go somewhere that has the bowl.
2. pick up the bowl.
3. go somewhere that has the desklamp.
4. use the desklamp.

### Environment information
{{basic_information}}

### Instruction
You are in planning mode. Using the given information, plan the full solution (overall path) so that you can solve the given task within the remaining steps.
Do NOT include any actions from the Full History. The history is already completed. Generate plans starting from the CURRENT state only (after the history). You go to a receptacle first before you can go, open, or close receptacle.
Generate {{num_plans}} valid plans.
For each plan, you must **precise** reason based on the given information. Then, generate the full solution (overall path) so that the given task is solved.
Based on this plan, you generate the full sequence of actions.
Each plan MUST have a full action sequence (at most remaining steps of actions).
Provide exactly {{num_plans}} plans labeled "Plan 1:" ,..., "Plan {{num_plans}}:".
If you think the task is done but observation is given, you can use it what is missing and what things must be done. Your plan MUST BE generate after the given history. IF you cool, heat, clean, you MUST select cool, heat, clean, **NOT the OTHER ACTION, such as move**.
"""

GRAPH_STEP1_SYSTEM = \
f"""Simulate a task in household environment, produce per-plan step sequences and generate the graph.

The available actions for household environment is:
1. go to {{{{recep}}}}
2. take {{{{obj}}}} from {{{{recep}}}}
3. move {{{{obj}}}} to {{{{recep}}}}
4. open {{{{recep}}}}
5. close {{{{recep}}}}
6. toggle {{{{obj}}}} {{{{recep}}}}
7. clean {{{{obj}}}} with {{{{recep}}}}
8. heat {{{{obj}}}} with {{{{recep}}}}
9. cool {{{{obj}}}} with {{{{recep}}}}
10. use {{{{obj}}}}
where {{{{obj}}}} and {{{{recep}}}} correspond to objects and receptacles. You MUST CONSIDER THE INDEX for {{{{obj}}}} as well as {{{{recep}}}}.

Here is the known information about the receptacle and object.
{{initial_info}}

## Instructions
- Do NOT include any nodes or actions from the history. The history is already done. Start the plan sequence from the CURRENT step only. The first node should represent the current state AFTER the history.
- Simulate each plan step-by-step using the household information.
- Build each plan's step sequence as alternating entries: node -> action -> node -> action -> ...
- A node entry must include a node id, a short thought, and the predicted observation object.
- Node ids must be unique across all plans (no duplicate node_id between plans).
- The thought must briefly explain how the previous action changes the observation.
- The observation must be an object with keys: inventory, your_location, observation, subgoal_progress. For inventory in the observation, it is the list of objects such as ({{{{"object": "object_name", "status": "heat/cold/none"}}}}) Initial states has no objects, so it is []. If you take some object, you must add that object in the inventory. If you move back some object, you must delete that object in the inventory. 
If you take the go action, you must update your location. your updated location must be matched with the receptacle you go to.
Observation contains what you see in the environment, the effect of your previous actions.
Subgoal progress contains information about the progress towards goals and the remaining steps.
- An action entry must include only the action. No separate thought key in action entries.
- Preserve ALL actions from every plan (do not drop steps).
- JSON keys must appear in this order: plan_sequences.
Return JSON only:
{{{{
  "plan_sequences": [
    {{{{
      "plan_id": "plan1",
      "steps": [
        {{{{
          "node_id": "nodeX",
          "thought": "Initial states",
          "observation": {{{{"inventory": [], "your_location": "middle", "observation": "none", "observation": "your observation on the node. (e.g., You are in the middle of the room, on it.)", "subgoal_progress": "Comparing with the goal, you are at the beginning of the task."}}}}
        }}}},
        {{{{
          "action": "go to wardrobe 1"
        }}}},
        {{{{
          "node_id": "nodeY",
          "thought": "After go wardrobe 1, you're location is wardrobe 1.",
          "observation": {{{{"inventory": [], "your_location": "wardrobe 1",
           "observation": "Your observation on the node. (e.g., You are now at wardrobe 1. / Nothing happended, You cooled apple 1.)", "subgoal_progress": "Comparing with the goal, you arrived at wardrobe 1 where you can find clothes. cleaning and moving it to safe is remaining."}}}}
        }}}}
      ]
    }}}}
  ]
}}}}
"""

GRAPH_STEP2_INSTRUCTIONS = """## Instructions
- After all plan sequences are built, merge identical nodes when the observation text matches exactly (including indexes of the core objects).
- Preserve ALL actions from every plan (do not drop steps). (Note that **index** in the action MUST BE MAINTAIN.)
- Construct the full graph (nodes + edges) from the merged nodes. For generating edges, you MUST USE "kept_node" id for "from" and "to" node to ensure correct mapping.
- Generate a thought for each edge based on the transition between observations.
- Mark start nodes explicitly with "is_start": true in full_graph nodes.
- Mark goal nodes explicitly with "is_goal": true in full_graph nodes.
- JSON keys must appear in this order: reasoning, merge_log, full_graph.

Return JSON only:
{
  "reasoning": "overall reasoning for node merging and graph construction",
  "merge_log": [
    {
      "reason": "same observation text",
      "kept_node": "nodeX",
      "merged_nodes": ["nodeXX", ...]
    }
  ],
  "full_graph": {
    "nodes": [
      {
        "id": "nodeX",
        "observation": {"inventory": [], "your_location": "middle", "observation": "none", "observation": "(Your observation on the node. (e.g., You are now at wardrobe 1. / Nothing happended, You cool apple 1.)"},
        "subgoal_progress": "Comparing with the goal, you arrived at wardrobe 1 where you can find clothes. cleaning and moving it to safe is remaining.",
        "is_start": true or false,
        "is_goal": true or false
      }
    ],
    "edges": [
      {
        "from": "nodeX",
        "to": "nodeY",
        "thought": "why this step is taken",
        "action": "go to wardrobe 1"
      }
    ]
  }
}
"""

PLAN_USER_TEMPLATE = \
f"""### Task
{{task}}
### Full History before the current step
{{observation}}
"""

GRAPH_STEP1_USER_TEMPLATE = \
f"""### Task
{{task}}
### Full History before the current step
{{observation}}
# Plans text After the current step
{{plans_text}}"""

GRAPH_STEP2_USER_TEMPLATE = \
f"""Use the plan_sequences you just produced.
{{instructions}}"""

# Consider the given information of the environment to assign the score: {{basic_information}}
SCORE_NODES_SYSTEM = \
f"""You score all the states in the graph. Higher score means closer to solving.
Assign reward 1.0 to goal states (Given goal is done). Assign score -1.0 if, starting from this state, there is no valid sequence of actions that can ever reach any goal state (deadlocked/unreachable). All other states (i.e., remaining few steps to accomplish the goal) should have score 0.0. 

**Use the Available Transitions (Edges) to determine reachability.** A state is unreachable (-1.0) if there is no path through the edges that leads to a goal state.

If you want to heat something, only heating object with microwave is the option. If you want to cool, only cooling object with fridge is the option. If you want to clean, only cleaning object at sinkbasin is the option. If you need to cool but not use fridge, or heat but not use microwave, or clean but not use sinkbasin, then **assign score -1.0** (no other way to achieve the goal).

Only use scores -1.0, 0.0, or 1.0 (no other values).
For each state, provide a short reasoning sentence before assigning its score.
Return JSON only with both reasons and scores, for example:
{{{{
  "reasons": {{{{
    "s0": "reasoning for s0: whether any sequence can solve the household task based on available transitions.",
    ...
  }}}},
  "scores": {{{{
    "s0": 0.0,
    ...
  }}}}
}}}}.
"""

SCORE_NODES_USER_TEMPLATE = f"""### Goal
{{task}}
### Your Current Observation (Summary)
{{observation}}
### Available Transitions (Edges)
{{edges}}
### Target States to Score
{{states}}"""

OBSERVATION_EXTRACT_SYSTEM = """You extract the current environment observation from a full conversation history (Observation, Thought, and Action.). Based on the history, You determine what you are taken in the inventory, where is your current location, and which observation is done and remaining. 
Return only JSON that matches the schema. Use inventory items with status in {"heat","cold","none"}. Note that the initial state has **no items** in the inventory (i.e., empty list []).
If a field is unknown, return an empty string or empty list.
FORMAT AS FOLLOWS:
{
  "inventory": [
    {"object": "object", "status": "heat/cold/none"},
    ...
  ],
  "your_location": "your location.",
  "observation": "Your observation on the node. (e.g., You are now at wardrobe 1. / Nothing happended, You cool apple 1.)",
  "subgoal_progress": "Your progress towards the goal. (e.g., Comparing with the goal, you arrived at wardrobe 1 where you can find clothes. cleaning and moving it to safe is remaining.)"
}"""

OBSERVATION_EXTRACT_USER_TEMPLATE = """History:
{history}"""

STATE_EQUAL_SYSTEM = """Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. You need to determine if two states are equal or not.

Compare two environment states by focusing on:
1. Inventory items: Do they contain the same objects with the same statuses?
2. Location: Are both states in the same location?
3. Action outcome: Did the previous action succeed or fail? (e.g., "Nothing happened" means failure, status changes like "heated", "cooled", "cleaned" mean success, receptacle opened/closed, object picked up, etc.)

**IMPORTANT: Do NOT compare which objects are visible in the observation. Only compare whether the previous action succeeded or failed and what effect it had.**

Return only JSON:
{
  "reason": "reasoning for equality/inequality based on action outcome",
  "is_equal": true/false
}"""

STATE_EQUAL_USER_TEMPLATE = """State A:
{state_a}

State B:
{state_b}"""

SYSTEM_PROMPT_TEMPLATE = """
{{ INSTRUCTION }}

{% if examples_text %}
{{ icl_prompt }}
{{ examples_text }}
{% endif %}
"""



"""Prompts specific to PA agent."""

PA_REACT_INSTRUCTION = \
f"""Interact with Sokoban environment to solve a task (placing every box onto goals.)

{{sokoban_rule}}

## Instruction
You are the player in a Sokoban environment, and your goal is to place every box on a goal within a limited number of actions (within step remaining). That means you need to make the same number of boxes on goals by placing all boxes onto goals. If a box is on a goal, it is considered satisfied.

At each turn, you will receive the current observation.
Observation format (all coordinates are (x, y)):
- wall location: (x1, y1), (x2, y2), ...
- player location: (x, y)
- box location: (x3, y3), ...
- goal location: (x4, y4), ...
- box on goal location: (x5, y5), ...
- Step remaining: <steps_remaining>
The "box location" list includes only boxes not on goals. The "goal location" list includes all goals.
You may choose between two outputs: "Thought" or "Action".

When you choose "Thought", you must:
The initial full plan is given, so you compare the current observation and the full plan to identify the current state in the given plan and follow the given plan if it matches the current observation.
If there is a mismatch between plan and observation, you can replan and generate the action as follows:
Replan the full remaining path so that all boxes reach goals within the remaining steps, using the Sokoban rules (task + mechanics). From that full plan, explicitly predict only the next 1 step: how the immediate action will change the player location, box location, and box on goal location. Based on that planning and the current observation, decide the immediate next action to take.
IMPORTANT:
    - Your Thought MUST match the given observation exactly. No hallucination about positions or adjacency is allowed.
    - If a push is feasible immediately, you MUST choose the move that pushes.
    - If you planned a push in the previous Thought and the current observation still allows that push, you MUST continue and execute it.
    - Do NOT rewrite the plan from scratch unless the environment changed and the old plan became infeasible.

Your output must follow exactly:
Thought: <your reasoning>
Action: <U|D|L|R>

Here is the initial full plan:
{{plan_text}}
"""

PA_PLAN_INSTRUCTION = \
f"""Interact with Sokoban environment to solve a task (placing every box onto goals.)

{{sokoban_rule}}

## Instruction
You are the player in a Sokoban environment, and your goal is to place every box on a goal within a limited number of actions (within step remaining). That means you need to make the same number of boxes on goals by placing all boxes onto goals. If a box is on a goal, it is considered satisfied.

At each turn, you will receive the current observation.
Observation format (all coordinates are (x, y)):
- wall location: (x1, y1), (x2, y2), ...
- player location: (x, y)
- box location: (x3, y3), ...
- goal location: (x4, y4), ...
- box on goal location: (x5, y5), ...
- Step remaining: <steps_remaining>
The "box location" list includes only boxes not on goals. The "goal location" list includes all goals.

You are in planning mode. Using the Sokoban rules, plan the full solution (overall path) so that all boxes reach goals within the remaining steps, using the Sokoban rules (task + mechanics).
Based on this plan, you generate the full sequence of actions."""


PA_PLAN_INSTRUCTION_ARITHMETIC = """Solve the arithmetic question with the provided tools under budget constraints.

## Instruction
You are in planning mode.
Your goal is to decide which tools to use, in what order, and with what arguments.

Rules:
1. Plan using only the provided tools.
2. Consider each tool's output quality and execution-time distribution.
3. Minimize risk of budget overrun while preserving answer reliability.
4. Do not use internal knowledge or mental math for the final answer.
5. If the tool outputs may be insufficient, your plan should include an Unknown fallback.

Available arithmetic tools:
{tool_descriptions}

Output a concise executable plan as numbered steps.
Each step should specify:
- intended tool
- intended inputs
- purpose of the step
- stop condition for issuing final answer

Planning mode output does not need strict JSON.
Use plain numbered text steps."""


PA_REACT_INSTRUCTION_ARITHMETIC = """Solve the arithmetic question with tools by following the initial plan first.

## Instruction
At each turn, you may:
1. Call one tool (structured JSON format, preferred):
{{"thought": "...", "tool_name": "<tool>", "arguments": {{"a": 1, "b": 2}}}}

2. Provide final answer:
Thought: <reasoning>
Action: Answer is \\boxed{{ANS}}.

Rules:
1. Follow the initial plan when possible.
2. If observations/tool results contradict the plan, replan briefly and continue.
3. Use only information returned by tools in this episode.
4. Do not use internal knowledge, guessing, or mental math.
5. If the answer cannot be derived from available tool outputs, output Unknown:
Action: Answer is \\boxed{{Unknown}}.

Available arithmetic tools:
{tool_descriptions}

Here is the initial full plan:
{plan_text}
"""

PA_PLAN_INSTRUCTION_ALFWORLD = \
f"""Interact with a household to solve a task within the given steps.

Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal.

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

Here are the examples for high level action sequences you may take:
Task Type 1: put some spraybottle on toilet
High level actions:
1. go somewhere that has the spraybottle.
2. go to the spraybottle.
3. go to the toilet.
4. move the spraybottle to the toilet.

Task Type 2: put a clean lettuce in diningtable.
High level actions:
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
High level actions:
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
4. use the desklamp.

### Instruction
You are in planning mode. Using the given information, plan the full solution (overall path) so that you can solve the given task within the remaining steps.
Based on this plan, you generate the full sequence of actions.
"""

PA_REACT_INSTRUCTION_ALFWORLD = \
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
High level actions:
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
High level actions:
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
4. use the desklamp.

Here is the initial full plan:
{{plan_text}}
"""

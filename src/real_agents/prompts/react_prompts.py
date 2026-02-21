"""Prompts for ReACT/PA agents."""

RULE = \
"""## Sokoban rules (task + mechanics)
- Task objective: place every box onto goals. The puzzle is solved when all boxes are on goals.
- Action mechanics: each action moves the player exactly one cell in the chosen direction (U/D/L/R).
    - U: move up, e.g., (x, y) -> (x, y + 1)
    - D: move down, e.g., (x, y) -> (x, y - 1)
    - L: move left, e.g., (x, y) -> (x - 1, y)
    - R: move right, e.g., (x, y) -> (x + 1, y)
    - The action name must match the coordinate change (U increases y, D decreases y, R increases x, L decreases x).
- Walls: the player cannot move into a wall cell. If a wall occupies the destination cell, the action is invalid and the player does not move.
- Boxes: if the destination cell has a box, the player attempts to push it one cell further in the same direction.
    - The push succeeds only if the cell behind the box is empty floor or a goal.
    - If the cell behind the box is a wall or another box, the push is invalid and nothing moves.
    - A push is only possible when the player is on the cell immediately adjacent to the box from the opposite side of the push direction.
- The player cannot pull boxes and cannot push two boxes at once.
- Some pushes can create deadlocks (for example, pushing a box into a corner where it cannot reach any goal).

Examples (coordinate outcomes, using U/D/L/R only):
- Empty move, R: player at (x, y), no wall/box at (x + 1, y). Action R -> player at (x + 1, y); box on goal location unchanged (goals are unaffected by moves).
- Empty move, L: player at (x, y), no wall/box at (x - 1, y). Action L -> player at (x - 1, y); box on goal location unchanged (goals are unaffected by moves).
- Empty move, U: player at (x, y), no wall/box at (x, y + 1). Action U -> player at (x, y + 1); box on goal location unchanged (goals are unaffected by moves).
- Empty move, D: player at (x, y), no wall/box at (x, y - 1). Action D -> player at (x, y - 1); box on goal location unchanged (goals are unaffected by moves).
- Wall block, U: player at (x, y), wall at (x, y + 1). Action U -> invalid, player stays at (x, y); box on goal location unchanged.
- Wall block, D: player at (x, y), wall at (x, y - 1). Action D -> invalid, player stays at (x, y); box on goal location unchanged.
- Wall block, L: player at (x, y), wall at (x - 1, y). Action L -> invalid, player stays at (x, y); box on goal location unchanged.
- Wall block, R: player at (x, y), wall at (x + 1, y). Action R -> invalid, player stays at (x, y); box on goal location unchanged.
- Push succeeds, R: player at (x, y), box at (x + 1, y), no wall/box at (x + 2, y). Action R -> player at (x + 1, y), box moves to (x + 2, y); if (x + 2, y) is a goal, box on goal location becomes (x + 2, y), otherwise unchanged.
- Push succeeds, L: player at (x, y), box at (x - 1, y), no wall/box at (x - 2, y). Action L -> player at (x - 1, y), box moves to (x - 2, y); if (x - 2, y) is a goal, box on goal location becomes (x - 2, y), otherwise unchanged.
- Push succeeds, U: player at (x, y), box at (x, y + 1), no wall/box at (x, y + 2). Action U -> player at (x, y + 1), box moves to (x, y + 2); if (x, y + 2) is a goal, box on goal location becomes (x, y + 2), otherwise unchanged.
- Push succeeds, D: player at (x, y), box at (x, y - 1), no wall/box at (x, y - 2). Action D -> player at (x, y - 1), box moves to (x, y - 2); if (x, y - 2) is a goal, box on goal location becomes (x, y - 2), otherwise unchanged.
- Push blocked by wall, R: player at (x, y), box at (x + 1, y), wall at (x + 2, y). Action R -> invalid, player stays at (x, y), box stays at (x + 1, y); box on goal location unchanged.
- Push blocked by wall, L: player at (x, y), box at (x - 1, y), wall at (x - 2, y). Action L -> invalid, player stays at (x, y), box stays at (x - 1, y); box on goal location unchanged.
- Push blocked by wall, U: player at (x, y), box at (x, y + 1), wall at (x, y + 2). Action U -> invalid, player stays at (x, y), box stays at (x, y + 1); box on goal location unchanged.
- Push blocked by wall, D: player at (x, y), box at (x, y - 1), wall at (x, y - 2). Action D -> invalid, player stays at (x, y), box stays at (x, y - 1); box on goal location unchanged.
- Push blocked by box, R: player at (x, y), box at (x + 1, y), another box at (x + 2, y). Action R -> invalid, player stays at (x, y), boxes stay at (x + 1, y) and (x + 2, y); box on goal location unchanged.
- Push blocked by box, L: player at (x, y), box at (x - 1, y), another box at (x - 2, y). Action L -> invalid, player stays at (x, y), boxes stay at (x - 1, y) and (x - 2, y); box on goal location unchanged.
- Push blocked by box, U: player at (x, y), box at (x, y + 1), another box at (x, y + 2). Action U -> invalid, player stays at (x, y), boxes stay at (x, y + 1) and (x, y + 2); box on goal location unchanged.
- Push blocked by box, D: player at (x, y), box at (x, y - 1), another box at (x, y - 2). Action D -> invalid, player stays at (x, y), boxes stay at (x, y - 1) and (x, y - 2); box on goal location unchanged.
"""

REACT_INSTRUCTION = \
f"""Interact with Sokoban environment to solve a task (placing every box onto goals.)

{RULE}

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
Plan the full solution (overall path) so that all boxes reach goals within the remaining steps, using the Sokoban rules (task + mechanics).
From that full plan, explicitly predict only the next 1 step: how the immediate action will change the player location, box location, and box on goal location.
Based on that planning and the current observation, decide the immediate next action to take.
IMPORTANT:
    - Your Thought MUST match the given observation exactly. No hallucination about positions or adjacency is allowed.
    - If a push is feasible immediately, you MUST choose the move that pushes.
    - If you planned a push in the previous Thought and the current observation still allows that push, you MUST continue and execute it.
    - Do NOT rewrite the plan from scratch unless the environment changed and the old plan became infeasible.

Your output must follow exactly:
Thought: <your reasoning>
Action: <U|D|L|R>"""

SYSTEM_PROMPT_TEMPLATE = """
{{ INSTRUCTION }}

{% if examples_text %}
{{ icl_prompt }}
{{ examples_text }}
{% endif %}
"""

REACT_INSTRUCTION_ALFWORLD = \
"""Interact with a household to solve a task within the given steps. 

## Instructions
Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:
"Thought: your thoughts.
Action: your next action"; 
If you choose "Action", you should directly output the action in this turn.
Your output must strictly follow this format:"Action: your next action".
The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. move {obj} to {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
10. use {obj}
where {obj} and {recep} correspond to objects and receptacles. You MUST CONSIDER THE INDEX for {obj} as well as {recep}.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.
Reminder: 
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

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
"""

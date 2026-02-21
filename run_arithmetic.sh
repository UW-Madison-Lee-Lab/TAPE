#!/bin/bash
agent=${1:-react}
model=${2:-gpt-4.1-mini}

TIME_CONSTRAINTS=(0.6 0.8 1.0 1.5 1.8 2.0 2.5 3.0)
COST_CONSTRAINTS=(0.02 0.04 0.02 0.04 0.05 0.07 0.06 0.1)

python src/arithmetic_experiment.py \
    --agent "$agent" \
    --model "$model" \
    --episodes 1 \
    --num_jobs 10 \
    --print_episodes -1 \
    --time_constraints ${TIME_CONSTRAINTS[@]} \
    --cost_constraints ${COST_CONSTRAINTS[@]}

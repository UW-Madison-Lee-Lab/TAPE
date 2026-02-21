agent=${1:-react}

python sokoban_real_experiments.py \
    --T_targets 10 \
    --print_episodes -1 \
    --num_jobs 10 \
    --model gpt-4.1 \
    --agent pa \
    --episodes 10 \
    --slack 2

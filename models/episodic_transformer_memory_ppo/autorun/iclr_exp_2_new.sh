#!/bin/bash

SESSION_NAME="iclr_exp_2_new_gtrxl"
seeds=(123 213 312)
REPO_PATH="/home/jovyan/Egor_C/REPOSITORIES/Memory-RL-Codebase"

tmux new-session -d -s $SESSION_NAME

for SEED in "${seeds[@]}"; do

    # K=8_L=21_random
    WINDOW_NAME="K=8_L=21_random_seed_${SEED}"
    tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "conda activate /home/jovyan/.mlspace/envs/pudge" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "python3 ../train.py \
    --environment_name 'MiniGrid-MemoryS13Random-v0' \
    --environment_length 21 \
    --log_name 'MinigridMemory/GTXL/iclr_exp_2_new/K=8_L=21_random' \
    --device 'cuda:0' \
    --eval_episodes 100 \
    --save_model True \
    --save_model_frequency 30 \
    --save_best_model True \
    --updates 100000000 \
    --epochs 5 \
    --transformer_memory_length 8 \
    --deterministic_torch False \
    --seed $SEED" C-m

    # K=21_L=21_random
    WINDOW_NAME="K=21_L=21_random_seed_${SEED}"
    tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "conda activate /home/jovyan/.mlspace/envs/pudge" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "python3 ../train.py \
    --environment_name 'MiniGrid-MemoryS13Random-v0' \
    --environment_length 21 \
    --log_name 'MinigridMemory/GTXL/iclr_exp_2_new/K=21_L=21_random' \
    --device 'cuda:0' \
    --eval_episodes 100 \
    --save_model True \
    --save_model_frequency 30 \
    --save_best_model True \
    --updates 100000000 \
    --epochs 5 \
    --transformer_memory_length 21 \
    --deterministic_torch False \
    --seed $SEED" C-m

    # K=8_L=21_fixed
    WINDOW_NAME="K=8_L=21_fixed_seed_${SEED}"
    tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "conda activate /home/jovyan/.mlspace/envs/pudge" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m

    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "python3 ../train.py \
    --environment_name 'MiniGrid-MemoryS13-v0' \
    --environment_length 21 \
    --log_name 'MinigridMemory/GTXL/iclr_exp_2_new/K=8_L=21_fixed' \
    --device 'cuda:0' \
    --eval_episodes 100 \
    --save_model True \
    --save_model_frequency 30 \
    --save_best_model True \
    --updates 100000000 \
    --epochs 5 \
    --transformer_memory_length 8 \
    --deterministic_torch False \
    --seed $SEED" C-m

    # K=21_L=21_fixed
    WINDOW_NAME="K=21_L=21_fixed_seed_${SEED}"
    tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PYTHONPATH=\$PYTHONPATH:\$REPO_PATH" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "conda activate /home/jovyan/.mlspace/envs/pudge" C-m
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" "export PATH=\$PATH:\$REPO_PATH" C-m

    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "python3 ../train.py \
    --environment_name 'MiniGrid-MemoryS13-v0' \
    --environment_length 21 \
    --log_name 'MinigridMemory/GTXL/iclr_exp_2_new/K=21_L=21_fixed' \
    --device 'cuda:0' \
    --eval_episodes 100 \
    --save_model True \
    --save_model_frequency 30 \
    --save_best_model True \
    --updates 100000000 \
    --epochs 5 \
    --transformer_memory_length 21 \
    --deterministic_torch False \
    --seed $SEED" C-m

done

tmux attach-session -t $SESSION_NAME

# seeds = (123 213 321)

# ### iclr_exp_2_new_gtrxl

# # K=8_L=21_random

# python script_name.py \
#     --environment_name "MiniGrid-MemoryS13Random-v0" \
#     --environment_length 21 \
#     --log_name "MinigridMemory/MinigridMemory/GTXL/iclr_exp_2_new/K=8_L=21_random" \
#     --device "cuda:0" \
#     --eval_episodes 100 \
#     --save_model True \
#     --save_model_frequency 3 \
#     --save_best_model True \
#     --updates 100000000 \
#     --epochs 5 \
#     --transformer_memory_length 8 \
#     --deterministic_torch False \
#     --seed 123 \

# # K=21_L=21_random

# python script_name.py \
#     --environment_name "MiniGrid-MemoryS13Random-v0" \
#     --environment_length 21 \
#     --log_name "MinigridMemory/MinigridMemory/GTXL/iclr_exp_2_new/K=21_L=21_random" \
#     --device "cuda:0" \
#     --eval_episodes 100 \
#     --save_model True \
#     --save_model_frequency 3 \
#     --save_best_model True \
#     --updates 100000000 \
#     --epochs 5 \
#     --transformer_memory_length 21 \
#     --deterministic_torch False \
#     --seed 123 \

# # K=8_L=21_fixed

# python script_name.py \
#     --environment_name "MiniGrid-MemoryS13-v0" \
#     --environment_length 21 \
#     --log_name "MinigridMemory/MinigridMemory/GTXL/iclr_exp_2_new/K=8_L=21_fixed" \
#     --device "cuda:0" \
#     --eval_episodes 100 \
#     --save_model True \
#     --save_model_frequency 3 \
#     --save_best_model True \
#     --updates 100000000 \
#     --epochs 5 \
#     --transformer_memory_length 8 \
#     --deterministic_torch False \
#     --seed 123 \

# # K=21_L=21_fixed


# python script_name.py \
#     --environment_name "MiniGrid-MemoryS13-v0" \
#     --environment_length 21 \
#     --log_name "MinigridMemory/MinigridMemory/GTXL/iclr_exp_2_new/K=21_L=21_fixed" \
#     --device "cuda:0" \
#     --eval_episodes 100 \
#     --save_model True \
#     --save_model_frequency 3 \
#     --save_best_model True \
#     --updates 100000000 \
#     --epochs 5 \
#     --transformer_memory_length 21 \
#     --deterministic_torch False \
#     --seed 123 \
















# python script_name.py \
#     --environment_type "MinigridMemory" \
#     --environment_name "MiniGrid-MemoryS13Random-v0" \
#     --environment_length 21 \
#     --environment_start_seed 0 \
#     --environment_num_seeds 100000 \
#     --environment_agent_scale 0.25 \
#     --environment_cardinal_origin_choice "[0,1,2,3]" \
#     --environment_show_origin False \
#     --environment_show_goal False \
#     --environment_visual_feedback True \
#     --environment_reward_goal 1.0 \
#     --environment_reward_fall_off 0.0 \
#     --environment_reward_path_progress 0.0 \
#     --project_name "online_transformers" \
#     --type "MinigridMemory" \
#     --log_name "MinigridMemory/MinigridMemory/GTXL/ICLR_exp_2_paper_random" \
#     --logger "tensorboard" \
#     --device "cuda:0" \
#     --eval_episodes 50 \
#     --save_model True \
#     --save_model_frequency 3 \
#     --save_best_model True \
#     --gamma 0.995 \
#     --lamda 0.95 \
#     --updates 100000000 \
#     --epochs 5 \
#     --n_workers 16 \
#     --worker_steps 512 \
#     --n_mini_batch 8 \
#     --value_loss_coefficient 0.5 \
#     --hidden_layer_size 128 \
#     --max_grad_norm 0.5 \
#     --transformer_embed_per_obs_dim 8 \
#     --transformer_num_blocks 6 \
#     --transformer_embed_dim 128 \
#     --transformer_num_heads 8 \
#     --transformer_memory_length 9 \
#     --transformer_positional_encoding "relative" \
#     --transformer_layer_norm "pre" \
#     --transformer_gtrxl True \
#     --transformer_gtrxl_bias 0.0 \
#     --lr_initial 3.5e-4 \
#     --lr_final 1.0e-4 \
#     --lr_power 1.0 \
#     --lr_max_decay_steps 250 \
#     --beta_initial 0.001 \
#     --beta_final 0.001 \
#     --beta_power 1.0 \
#     --beta_max_decay_steps 10000 \
#     --clip_initial 0.1 \
#     --clip_final 0.1 \
#     --clip_power 1.0 \
#     --clip_max_decay_steps 10000
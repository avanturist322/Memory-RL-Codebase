#!/bin/bash


SEEDS_LIST=(42 1337 123 22) 

SESSION_NAME="ICLR_exp_3_LSTM_DQN" 

PREFIX="run"  

tmux new-session -d -s $SESSION_NAME

for i in "${!SEEDS_LIST[@]}"; do
    SEED=${SEEDS_LIST[$i]}

    WINDOW_NAME="${PREFIX}_${i}"
    tmux new-window -t $SESSION_NAME -n $WINDOW_NAME

    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "python3 main.py --config_env configs/envs/tmaze_passive.py --config_env.env_name 58 --config_rl configs/rl/dqn_default.py --train_episodes 100000 --config_seq configs/seq_models/lstm_default.py --config_seq.sampled_seq_len 60 --seed ${SEED} --gpu_id 0 --save_dir 'logs/Passive_T_Maze_Flag/LSTM_DQN/ICLR_exp_3'" C-m
done

tmux attach-session -t $SESSION_NAME
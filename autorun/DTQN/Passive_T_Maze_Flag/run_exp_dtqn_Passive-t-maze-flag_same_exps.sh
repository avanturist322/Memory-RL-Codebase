#!/bin/bash

config_path="/opt/Memory-RL-Codebase/configs/DTQN_configs/Passive_T_Maze_Flag/Exp_for_Egor/Passive_T_Maze_Flag_SHORT_TERM_easy.yaml"
session_name="dtqn_passive_t_maze_flag_easy_T=200_L=10_M=90"
NUM_ITERS=1

tmux new-session -s $session_name -d
export PYTHONPATH=$PYTHONPATH:/opt/Memory-RL-Codebase
tmux_prefix="DTQN"

for i in $(seq 1 $NUM_ITERS); do
    config_name=$(basename "$config_path" .yaml)
    
    tmux_window_name="${tmux_prefix}_${config_name}_${i}"
    tmux new-window -t $session_name -n "$tmux_window_name"

    tmux send-keys -t $session_name:"$tmux_window_name" "export PYTHONPATH=\$PYTHONPATH:/opt/Memory-RL-Codebase" C-m
    tmux send-keys -t $session_name:"$tmux_window_name" "python3 /opt/Memory-RL-Codebase/models/DTQN/run.py --config $config_path" C-m
done

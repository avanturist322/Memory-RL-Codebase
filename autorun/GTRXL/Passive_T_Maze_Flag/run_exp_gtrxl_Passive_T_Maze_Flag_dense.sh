#!/bin/bash

config_list=(
    "/opt/Memory-RL-Codebase/configs/GTRXL_configs/Passive_T_Maze_Flag/Dense/Passive_T_Maze_Flag_LONG_TERM.yaml"
    "/opt/Memory-RL-Codebase/configs/GTRXL_configs/Passive_T_Maze_Flag/Dense/Passive_T_Maze_Flag_SHORT_TERM.yaml"
)
session_name="GTRXL_passive_t_maze_flag_dense4_emb_256"


tmux new-session -s $session_name -d
export PYTHONPATH=$PYTHONPATH:/opt/Memory-RL-Codebase
tmux_prefix="GTRXL"

for config_path in "${config_list[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    
    tmux_window_name="${tmux_prefix}_${config_name}"
    tmux new-window -t $session_name -n "$tmux_window_name"

    tmux send-keys -t $session_name:"$tmux_window_name" "export PYTHONPATH=\$PYTHONPATH:/opt/Memory-RL-Codebase" C-m
    tmux send-keys -t $session_name:"$tmux_window_name" "python3 /opt/Memory-RL-Codebase/models/episodic_transformer_memory_ppo/train.py --config $config_path" C-m
done

#!/bin/bash

config_list=(
    "/opt/Memory-RL-Codebase/configs/TRXL_configs/Bsuite/MemoryLength/MemoryLengthEasy_LONG_TERM.yaml"
    "/opt/Memory-RL-Codebase/configs/TRXL_configs/Bsuite/MemoryLength/MemoryLengthEasy_SHORT_TERM.yaml"
    "/opt/Memory-RL-Codebase/configs/TRXL_configs/Bsuite/MemoryLength/MemoryLengthMedium_LONG_TERM.yaml"
    "/opt/Memory-RL-Codebase/configs/TRXL_configs/Bsuite/MemoryLength/MemoryLengthMedium_SHORT_TERM.yaml"
    "/opt/Memory-RL-Codebase/configs/TRXL_configs/Bsuite/MemoryLength/MemoryLengthHard_LONG_TERM.yaml"
    "/opt/Memory-RL-Codebase/configs/TRXL_configs/Bsuite/MemoryLength/MemoryLengthHard_SHORT_TERM.yaml"
)
session_name="TRXL_memorylength"


tmux new-session -s $session_name -d
export PYTHONPATH=$PYTHONPATH:/opt/Memory-RL-Codebase
tmux_prefix="TRXL"

for config_path in "${config_list[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    
    tmux_window_name="${tmux_prefix}_${config_name}"
    tmux new-window -t $session_name -n "$tmux_window_name"

    tmux send-keys -t $session_name:"$tmux_window_name" "export PYTHONPATH=\$PYTHONPATH:/opt/Memory-RL-Codebase" C-m
    tmux send-keys -t $session_name:"$tmux_window_name" "python3 /opt/Memory-RL-Codebase/models/episodic-transformer-memory-ppo/train.py --config $config_path" C-m
done

#!/bin/bash

config_list=(
    "/opt/Memory-RL-Codebase/configs/DTQN_configs/MinigridMemory/Static/MinigridMemory_SHORT_TERM.yaml"
    #"/opt/Memory-RL-Codebase/configs/DTQN_configs/MinigridMemory/MinigridMemory_LONG_TERM.yaml"
)

session_name="dtqn_minigridmemory_static_context_60_run_1"


tmux new-session -s $session_name -d
export PYTHONPATH=$PYTHONPATH:/opt/Memory-RL-Codebase
tmux_prefix="DTQN"

for config_path in "${config_list[@]}"; do
    config_name=$(basename "$config_path" .yaml)
    
    tmux_window_name="${tmux_prefix}_${config_name}"
    tmux new-window -t $session_name -n "$tmux_window_name"

    tmux send-keys -t $session_name:"$tmux_window_name" "export PYTHONPATH=\$PYTHONPATH:/opt/Memory-RL-Codebase" C-m
    tmux send-keys -t $session_name:"$tmux_window_name" "python3 /opt/Memory-RL-Codebase/models/DTQN/run.py --config $config_path" C-m
done
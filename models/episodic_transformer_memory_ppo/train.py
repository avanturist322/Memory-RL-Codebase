import torch
# from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser
import argparse

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """


    parser = argparse.ArgumentParser(description="Training config")
    
    # run params
    parser.add_argument("--project_name", type=str, default="online_transformers")
    parser.add_argument("--type", type=str, default="MinigridMemory")
    parser.add_argument("--eval_static_and_random", type=bool, default=True) # for MinigridMemory only
    parser.add_argument("--log_name", type=str, default="MinigridMemory/MinigridMemory/GTXL/iclr_new_exp_test")
    parser.add_argument("--logger", type=str, default="tensorboard")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_episodes", type=int, default=50) # timestep eval: eval_episodes * num_workers * num_updates
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic_torch", type=bool, default=False)


    # env params
    parser.add_argument("--environment_type", type=str, default="MinigridMemory")
    parser.add_argument("--environment_name", type=str, default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--environment_length", type=int, default=21)
    parser.add_argument("--environment_start_seed", type=int, default=0)
    parser.add_argument("--environment_num_seeds", type=int, default=100000)
    parser.add_argument("--environment_agent_scale", type=float, default=0.25)
    parser.add_argument("--environment_cardinal_origin_choice", type=list, default=[0, 1, 2, 3])
    parser.add_argument("--environment_show_origin", type=bool, default=False)
    parser.add_argument("--environment_show_goal", type=bool, default=False)
    parser.add_argument("--environment_visual_feedback", type=bool, default=True)
    parser.add_argument("--environment_reward_goal", type=float, default=1.0)
    parser.add_argument("--environment_reward_fall_off", type=float, default=0.0)
    parser.add_argument("--environment_reward_path_progress", type=float, default=0.0)



    # logging params
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_model_frequency", type=int, default=3)
    parser.add_argument("--save_best_model", type=bool, default=True)

    # traning params
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--updates", type=int, default=100000000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--worker_steps", type=int, default=512)
    parser.add_argument("--n_mini_batch", type=int, default=8)
    parser.add_argument("--value_loss_coefficient", type=float, default=0.5)
    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # transformer params
    parser.add_argument("--transformer_embed_per_obs_dim", type=int, default=8)
    parser.add_argument("--transformer_num_blocks", type=int, default=6)
    parser.add_argument("--transformer_embed_dim", type=int, default=128)
    parser.add_argument("--transformer_num_heads", type=int, default=8)
    parser.add_argument("--transformer_memory_length", type=int, default=8)
    parser.add_argument("--transformer_positional_encoding", type=str, default="relative")
    parser.add_argument("--transformer_layer_norm", type=str, default="pre")
    parser.add_argument("--transformer_gtrxl", type=bool, default=True)
    parser.add_argument("--transformer_gtrxl_bias", type=float, default=0.0)

    # schedules params
    parser.add_argument("--learning_rate_schedule_initial", type=float, default=3.5e-4)
    parser.add_argument("--learning_rate_schedule_final", type=float, default=1.0e-4)
    parser.add_argument("--learning_rate_schedule_power", type=float, default=1.0)
    parser.add_argument("--learning_rate_schedule_max_decay_steps", type=int, default=250)

    parser.add_argument("--beta_schedule_initial", type=float, default=0.001)
    parser.add_argument("--beta_schedule_final", type=float, default=0.001)
    parser.add_argument("--beta_schedule_power", type=float, default=1.0)
    parser.add_argument("--beta_schedule_max_decay_steps", type=int, default=10000)

    parser.add_argument("--clip_range_schedule_initial", type=float, default=0.1)
    parser.add_argument("--clip_range_schedule_final", type=float, default=0.1)
    parser.add_argument("--clip_range_schedule_power", type=float, default=1.0)
    parser.add_argument("--clip_range_schedule_max_decay_steps", type=int, default=10000)
    args = parser.parse_args()

    config = {}
    

    prefixes = ['beta_schedule', 'clip_range_schedule', 'learning_rate_schedule', 'transformer', 'environment']

    for key, value in vars(args).items():
        for prefix in prefixes:
            if key.startswith(prefix + '_'):
                sub_name = key[len(prefix) + 1:]
                config.setdefault(prefix, {})[sub_name] = value
            else:
                config[key] = value


    device = torch.device(config['device'])
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor") ### fix this for images !!!

    print(config)
    
    trainer = PPOTrainer(config, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()
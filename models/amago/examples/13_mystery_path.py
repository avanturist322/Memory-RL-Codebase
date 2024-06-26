from argparse import ArgumentParser

import wandb

import amago


from amago.envs.builtin.mystery_path import MysteryPathGymEnv
#np.random.seed(seed)


from amago.cli_utils import *


def add_cli(parser):
    parser.add_argument("--env", type=str, default="MysteryPath-Grid-v0")
    parser.add_argument("--max_seq_len", type=int, default=4200)
    parser.add_argument("--traj_save_len", type=int, default=2000)
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--horizon", type=int, required=True)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        # no need to risk numerical instability when returns are this bounded
        # "amago.agent.Agent.reward_multiplier": 100.0,
    }
    turn_off_goal_conditioning(config)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        # NOTE: paper (and original POPGym results) use `memory_size=256`
        memory_size=args.memory_size,
        # NOTE: paper used layers=3
        layers=args.memory_layers,
    )
    # ! arch - 'ff'
    switch_tstep_encoder(config, arch="ff", n_layers=2, d_hidden=512, d_output=256)
    # switch_tstep_encoder(config, arch="cnn", 
    #                     #  n_layers=2, d_hidden=512, d_output=256, 
    #                      channels_first=True)
    # if args.naive:
    #     naive(config)
    use_config(config, args.configs)

    group_name = f"{args.run_name}_{args.env}"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"

        # make_train_env = lambda: POPGymEnv(f"popgym-{args.env}-v0")

        make_train_env = lambda: MysteryPathGymEnv(args.env, args.horizon) # "MysteryPath-Grid-v0"

        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            # For this one script to work across every environment,
            # these are arbitrary sequence limits we'll never each.
            max_seq_len=args.max_seq_len,
            traj_save_len=args.traj_save_len,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=1000 * 4,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        experiment.load_checkpoint(loading_best=True)
        experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        wandb.finish()


"""
 python3 models/amago/examples/13_mystery_path.py --env 'MysteryPath-Grid-v0' --parallel_actors 1 --trials 3 --epochs 200 --dset_max_size 5_000 --memory_layers 3 --memory_size 256 --gpu 0 --run_name amago_mystery_path_h128 --buffer_dir checkpoints/amago --ckpt_interval 20 --val_interval 20 --traj_encoder transformer --horizon 128 --traj_save_len 128 --max_seq_len 128
"""

# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
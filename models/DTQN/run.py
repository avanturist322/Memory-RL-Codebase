import os
import time
import argparse
import torch
import wandb
from utils import random, agent_utils, env_processing, logging_utils
from yaml_parser import YamlParser
from docopt import docopt
from dataclasses import dataclass


# envs
import gym, envs

try:
    import gym_pomdps
except ImportError:
    print(
        "WARNING: ``gym_pomdps`` is not installed. This means you cannot run an experiment with the HeavenHell or Hallway domain."
    )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DictWrapper:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError(f"'DictWrapper' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

def run_experiment(args):

    save_dir = os.path.join(os.getcwd(), "policies", args['project_name'], args['env'])
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(
        save_dir,
        f"model={args['model']}_env={args['env']}_obsembed={args['obsembed']}_inembed={args['inembed']}_context={args['context']}_heads={args['heads']}_layers={args['layers']}_"
        f"batch={args['batch']}_gate={args['gate']}_identity={args['identity']}_history={args['history']}_pos={args['pos']}_seed={args['seed']}",
    )



    # If we have a checkpoint, load the checkpoint and resume training if more steps are needed.
    # Or exit early if we have already finished training.
    if args['eval_mode'] and os.path.exists(policy_path + "_mini_checkpoint.pt"):
        steps_completed = agent.load_mini_checkpoint(policy_path)["step"]
        print(
            f"Found a mini checkpoint that completed {steps_completed} training steps."
        )
        if steps_completed >= args['num_steps']:
            print(f"Removing checkpoint and exiting...")
            if os.path.exists(policy_path + "_checkpoint.pt"):
                os.remove(policy_path + "_checkpoint.pt")
            exit(0)
        else:
            wandb_kwargs = {"resume": "must", "id": agent.load_checkpoint(policy_path)}
    # Begin training from scratch
    else:
        wandb_kwargs = {"resume": None}

    # args = wandb.config

    start_timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    max_ret = -9999999999999 # just init with low number


    start_time = time.time()
    env = env_processing.make_env(args['env'], args)
    eval_env = env_processing.make_env(args['env'], args)
    device = torch.device(args['device'])
    random.set_global_seed(args['seed'], env, eval_env)

    agent = agent_utils.get_agent(
        args['model'],
        env,
        eval_env,
        args['obsembed'],
        args['inembed'],
        args['buf_size'],
        device,
        args['lr'],
        args['batch'],
        args['context'],
        args['history'],
        args['num_steps'],
        # DTQN specific
        args['heads'],
        args['layers'],
        args['dropout'],
        args['identity'],
        args['gate'],
        args['pos'],
    )

    #if not os.path.exists(policy_path + "_mini_checkpoint.pt"):
    agent.prepopulate(args['prepopulate_steps'])


    print(
        f"[ {logging_utils.timestamp()} ] Creating {args['model']} with {sum(p.numel() for p in agent.policy_network.parameters())} parameters"
    )

    args['policy_network_params'] = sum(p.numel() for p in agent.policy_network.parameters())

    if args['logger'] == 'tensorboard':
        logger = logging_utils.TensorboardLogger(config = args)
    elif args['logger'] == 'csv':
        logger = logging_utils.CSVLogger(policy_path)
    elif args['logger'] == 'wandb':
        logging_utils.wandb_init(
            args,
            [
                "model",
                "obsembed",
                "inembed",
                "context",
                "heads",
                "layers",
                "batch",
                "gate",
                "identity",
                "history",
                "pos",
            ],
            **wandb_kwargs,
        )
        logger = wandb


    # Enjoy mode
    if args['render']:
        agent.policy_network.load_state_dict(
            torch.load(policy_path, map_location="cpu")
        )
        agent.policy_network.eval()
        agent.exp_coef.val = 0
        while True:
            agent.evaluate(n_episode=1, render=True)
        exit(0)


    # Pick up from where we left off in the checkpoint (or 0 if doesn't exist) until max steps
    for timestep in range(agent.num_steps, args['num_steps']):
        agent.step()
        agent.train()

        if timestep % args['tuf'] == 0:
            agent.target_update()

        if timestep % args['eval_frequency'] == 0:
            sr, ret, length = agent.evaluate(args['eval_episodes'], args.get('eval_seeds', None))
            if agent.num_steps < len(agent.td_errors):
                td_error = agent.td_errors.sum() / agent.num_steps
                grad_norm = agent.grad_norms.sum() / agent.num_steps
                qmax = agent.qvalue_max.sum() / agent.num_steps
                qmean = agent.qvalue_mean.sum() / agent.num_steps
                qmin = agent.qvalue_min.sum() / agent.num_steps
                tmax = agent.target_max.sum() / agent.num_steps
                tmean = agent.target_mean.sum() / agent.num_steps
                tmin = agent.target_min.sum() / agent.num_steps
            else:
                td_error = agent.td_errors.mean()
                grad_norm = agent.grad_norms.mean()
                qmax = agent.qvalue_max.mean()
                qmean = agent.qvalue_mean.mean()
                qmin = agent.qvalue_min.mean()
                tmax = agent.target_max.mean()
                tmean = agent.target_mean.mean()
                tmin = agent.target_min.mean()

            if agent.num_evals < len(agent.episode_rewards):
                mean_reward = agent.episode_rewards.sum() / agent.num_evals
                mean_success_rate = agent.episode_successes.sum() / agent.num_evals
                mean_episode_length = agent.episode_lengths.sum() / agent.num_evals
            else:
                mean_reward = agent.episode_rewards.mean()
                mean_success_rate = agent.episode_successes.mean()
                mean_episode_length = agent.episode_lengths.mean()

            logger.log(
                {
                    "training/losses/TD_Error": td_error,
                    "training/losses/Grad_Norm": grad_norm,
                    "training/losses/Max_Q_Value": qmax,
                    "training/losses/Mean_Q_Value": qmean,
                    "training/losses/Min_Q_Value": qmin,
                    "training/losses/Max_Target_Value": tmax,
                    "training/losses/Mean_Target_Value": tmean,
                    "training/losses/Min_Target_Value": tmin,
                    "training/Mean_Return": mean_reward, # "results/Mean_Return": mean_reward,
                    "training/Mean_Success_Rate": mean_success_rate,
                    "training/Mean_Episode_Length": mean_episode_length,
                    "training/Hours": (time.time() - start_time) / 3600,
                    "evaluation/Return": ret,
                    "evaluation/Success_Rate": sr,
                    "evaluation/Episode_Length": length,
                },
                step=agent.num_steps,
            )

            if ret > max_ret and args['save_best_model']:
                max_ret = ret
                if not os.path.exists(os.path.join('../../checkpoints', args['log_name'], 'best_model')):
                    os.makedirs(os.path.join('../../checkpoints', args['log_name'], 'best_model'))

                agent.save_checkpoint('tensorboard', os.path.join('../../checkpoints', args['log_name'], 'best_model', logger.timestamp.strip('/') + '.pt'))
                
                print('best model checkpoint saved!')


            if args['verbose']:
                curtime = logging_utils.timestamp()
                print(
                    f"[ {curtime} ] Eval #{agent.num_evals} Success Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}, Hours: {((time.time() - start_time) / 3600):.2f}"
                )

        if args['save_model'] and timestep % args['save_model_frequency'] == 0:

            if not os.path.exists(os.path.join('../../checkpoints', args['log_name'])):
                os.makedirs(os.path.join('../../checkpoints', args['log_name']))
            agent.save_checkpoint('tensorboard', os.path.join('../../checkpoints', args['log_name'], logger.timestamp.strip('/') + '.pt'))
            print('checkpoint saved!')

        # if args['save_policy'] and not timestep % 50_000:
        #     torch.save(agent.policy_network.state_dict(), policy_path)

        # if (
        #     args['time_limit'] is not None
        #     and ((time.time() - start_time) / 3600) > args['time_limit']
        # ):
        #     print(
        #         f"Reached time limit. Saving checkpoint with {agent.num_steps} steps completed."
        #     )
        #     agent.save_checkpoint(wandb.run.id, policy_path)
        #     exit(0)
    # In case we finish before time limit, we need to output a mini checkpoint so as not to repeat ourselves
    # agent.save_mini_checkpoint(wandb_id=wandb.run.id, checkpoint_dir=policy_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/POPGym_RepeatPreviousEasy.yaml",
        help="The project name (for wandb) or directory name (for local logging) to store the results.",
    )

    config = YamlParser(parser.parse_args().config).get_config() #AttrDict()
    run_experiment(config)
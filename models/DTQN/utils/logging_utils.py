import wandb
import csv
import os
from typing import Dict
from datetime import datetime
import time 
from torch.utils.tensorboard import SummaryWriter
import yaml 



def timestamp():
    return datetime.now().strftime("%B %d, %H:%M:%S")

    # parser.add_argument(
    #     "--env", type=str, default="Memory-5-v0", help="Domain to use."
    # )

def get_log_env_name(init_env_name):
    if init_env_name == 'Memory-5-v0':
        env_name = 'MemoryCards_' 
    elif init_env_name in ['AutoencodeEasy', 'RepeatPreviousEasy', 'RepeatPreviousMedium', 'RepeatPreviousHard']:
        env_name = 'POPGym_'
    else:
        env_name = config["type"] + '_'

    return env_name

def wandb_init(config, group_keys, **kwargs) -> str:

    env_name = get_log_env_name(config['env'])

    wandb.init(
        project= env_name + config["project_name"],
        name = config["env"] + '_' + config["model"],
        group="_".join(
            [f"{key}={val}" for key, val in config.items() if key in group_keys]
        ),
        config=config,
        **kwargs,
    )

def dict_to_markdown_table(config_dict):
    table = "| Parameter | Value |\n| --- | --- |\n"
    for key, value in config_dict.items():
        table += f"| {key} | {value} |\n"
    return table


class TensorboardLogger:
    def __init__(self, config):
        self.log_name = config['log_name'] 

        if not os.path.exists("../../logs"):
            os.makedirs("../../logs")
        self.timestamp = time.strftime("/%Y_%m_%d-%H_%M_%S" + "/")
        self.writer = SummaryWriter("../../logs/" + self.log_name + self.timestamp)

        config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
        #yaml_config = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        markdown_config = dict_to_markdown_table(config_dict)


        self.writer.add_text("config", markdown_config, 0)



    def log(self, results: Dict[str, str], step: int):
        for key, value in results.items():
            self.writer.add_scalar(key, value, step)










class CSVLogger:
    """Logger to write results to a CSV. The log function matches that of Weights and Biases.

    Args:
        path: path for the csv results file
    """

    def __init__(self, path: str):
        self.results_path = path + "_results.csv"
        self.losses_path = path + "_losses.csv"
        # If we have a checkpoint, we don't want to overwrite
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Step",
                        "Success Rate",
                        "Return",
                        "Episode Length",
                        "Hours",
                        "Mean Success Rate",
                        "reward_mean",#"Mean Return",
                        "Mean Episode Length",
                    ]
                )
        if not os.path.exists(self.losses_path):
            with open(self.losses_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Step",
                        "TD Error",
                        "Grad Norm",
                        "Max Q Value",
                        "Mean Q Value",
                        "Min Q Value",
                        "Max Target Value",
                        "Mean Target Value",
                        "Min Target Value",
                    ]
                )

    def log(self, results: Dict[str, str], step: int):
        with open(self.results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    step,
                    results["results/Success_Rate"],
                    results["results/Return"],
                    results["results/Episode_Length"],
                    results["results/Hours"],
                    results["results/Mean_Success_Rate"],
                    results["results/Mean_Return"],
                    results["results/Mean_Episode_Length"],
                ]
            )
        with open(self.losses_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    step,
                    results["losses/TD_Error"],
                    results["losses/Grad_Norm"],
                    results["losses/Max_Q_Value"],
                    results["losses/Mean_Q_Value"],
                    results["losses/Min_Q_Value"],
                    results["losses/Max_Target_Value"],
                    results["losses/Mean_Target_Value"],
                    results["losses/Min_Target_Value"],
                ]
            )

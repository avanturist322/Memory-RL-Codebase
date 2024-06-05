from omegaconf import OmegaConf


def main(config = None):
    import sys
    sys.path.append('./')
    sys.path.append('../')
    import numpy as np
    import random
    import jax
    from src.trainers.trainers_control import ControlTrainer
    from tqdm import tqdm
    import numpy as np
    
    from tqdm.contrib.logging import logging_redirect_tqdm
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    task_to_trainer={
        'minigrid_pixel':ControlTrainer,
        'minigrid_onehot':ControlTrainer,
        'tmazev1':ControlTrainer,
        'tmazev2':ControlTrainer,
        'tmaze_ours':ControlTrainer,
        'memory_gym':ControlTrainer,
        'popgym':ControlTrainer,
        'procgen':ControlTrainer,
        'mujoco':ControlTrainer,
        'memorymaze':ControlTrainer,
    }

    logger.info("Available Backends:"+str(jax.devices()))
    with wandb.init(
        config=config,
        mode='online',
    ):
        config = wandb.config
        config = OmegaConf.create(config.as_dict())
        key=jax.random.PRNGKey(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        trainer_config=config.trainer
        env_config=config.task
    
        #Train the model
        kwargs={'global_args':config,'trainer_config':trainer_config,'env_config':env_config,
                'seed':config.seed,'key':key,'wandb_run':wandb.run}
        trainer=task_to_trainer[env_config['task']](**kwargs)
        pbar = tqdm(total=config.steps)
        step_count=0
        last_step_count=0
        try:
            with logging_redirect_tqdm():
                while True:
                    loss,metrics,step_count=trainer.step()
                    pbar.update(n=step_count-last_step_count)
                    last_step_count=step_count
                    if metrics is not None:
                        logger.info("Seed: "+str(config.seed)+" Steps: "+str(step_count)+" Metrics: "+str(metrics))
                        wandb.log({'seed':config.seed,**metrics},step=step_count)

                    if step_count>=config.steps:
                        break
        except Exception as e:
            # Log the exception using the logger
            logger.exception("An exception occurred: {}".format(e))
        pbar.close()
    
    wandb.finish()
    #Need to do something about logging val_metric
    exit()


if __name__=='__main__':
    import wandb
    import yaml
    config_path = './config/mujoco/arelit_sweep.yaml'
    project = "ReLiT"
    with open(config_path, 'r') as stream:
        sweep_config = yaml.safe_load(stream)
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, main, count=8)

    
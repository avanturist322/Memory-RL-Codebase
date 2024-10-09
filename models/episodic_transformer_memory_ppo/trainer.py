import numpy as np
import os
import pickle
import time
import torch

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter


from buffer import Buffer
from model import ActorCriticModel
from utils import batched_index_select, create_env, polynomial_decay, process_episode_info
from worker import Worker
#import wandb
import uuid
import yaml

def dict_to_markdown_table(config_dict):
    table = "| Parameter | Value |\n| --- | --- |\n"
    for key, value in config_dict.items():
        table += f"| {key} | {value} |\n"
    return table


class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # init wandb logger synced with tensorboard
        # model_name = '_GTrXL' if config['transformer']['gtrxl'] else '_TrXL'
        # wandb_name = config['environment']['name'] + model_name
        # wandb.init(project=config['environment']['type'] + '_online_transformers', name = wandb_name, config = config, sync_tensorboard=True, )

        # Set members
        self.config = config
        self.device = device
        if self.device != torch.device("cpu"):
            torch.set_default_device(self.device)
        self.run_id = run_id
        self.num_workers = config["n_workers"]
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks = config["transformer"]["num_blocks"]
        self.embed_dim = config["transformer"]["embed_dim"]

        # Setup Tensorboard Summary Writer
        # if not os.path.exists("./summaries"):
        #     os.makedirs("./summaries")
        # timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        # self.writer = SummaryWriter("./summaries/" + run_id + timestamp)


        self.log_name = config['log_name'] 

        if not os.path.exists("../../logs"):
            os.makedirs("../../logs")
        self.timestamp = time.strftime("/%Y_%m_%d-%H_%M_%S" + "/")
        self.writer = SummaryWriter("../../logs/" + self.log_name + self.timestamp)



        # Init dummy environment to retrieve action space, observation space and max episode length
        print("Step 1: Init eval environment")
        self.eval_env = create_env(self.config["environment"])
        observation_space = self.eval_env.observation_space
        self.action_space_shape = (self.eval_env.action_space.n,)
        self.max_episode_length = self.eval_env.max_episode_steps

        start_info = '\n' + '#' * 10 + f''''\n
        [ENVIROMENT INFO]:\n
        action_space_shape:{self.action_space_shape}\n
        observation_space_shape:{observation_space.obs_shape}\n
        observation_space_type:{observation_space.obs_type}\n
        max_episode_length:{self.max_episode_length}\n
        ''' + '#' * 10 + '\n'

        print(start_info)




        # Init buffer
        print("Step 2: Init buffer")
        print(observation_space)
        self.buffer = Buffer(self.config, observation_space, self.action_space_shape, self.max_episode_length, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, observation_space, self.action_space_shape, self.max_episode_length).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        config['model_params'] = sum(p.numel() for p in self.model.parameters())
        model_params = config['model_params']
        print(f'Model params: {model_params}')

        # Log config
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
        # yaml_config = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        markdown_config = dict_to_markdown_table(config_dict)

        self.writer.add_text("config", markdown_config, 0)

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.num_workers)]
        self.worker_ids = range(self.num_workers)
        self.worker_current_episode_step = torch.zeros((self.num_workers, ), dtype=torch.long)
        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        self.obs = np.zeros((self.num_workers,) + observation_space.obs_shape, dtype=np.float32)
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()
        # print(self.obs.shape)

        # Setup placeholders for each worker's current episodic memory
        self.memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
        # Generate episodic memory mask used in attention
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """         
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model. Only the final model is saved."""
        print("Step 6: Starting training using " + str(self.device))
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)
        max_episode_mean_reward = -99999999999

        for update in range(self.config["updates"]):
            # Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)
            
            print('SAMPLE TRANING DATA')
            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats, grad_info = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = process_episode_info(episode_infos)
            print(sampled_episode_info)

            # Print training statistics
            if "success" in episode_result:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            else:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            print(result)
            print(episode_result["reward_mean"], max_episode_mean_reward, self.config['save_best_model'])
            print(self.config['save_model'], update % self.config['save_model_frequency'] == 0)

            evaluation_stats = self._evaluate()

            # Write training statistics to tensorboard
            curr_step = update * self.config["epochs"] * self.config["n_mini_batch"] 


            self._write_gradient_summary(curr_step, grad_info)
            self._write_training_summary(curr_step, training_stats, episode_result)
            self._write_evaluation_summary(curr_step, evaluation_stats)


            if episode_result["reward_mean"] > max_episode_mean_reward and self.config['save_best_model']:
                max_episode_mean_reward = episode_result["reward_mean"]

                if not os.path.exists(os.path.join('../../checkpoints', self.config['log_name'], 'best_model')):
                    os.makedirs(os.path.join('../../checkpoints', self.config['log_name'], 'best_model'))

                self._save_model(os.path.join('../../checkpoints', self.config['log_name'], 'best_model', self.timestamp.strip('/') + '.pt'), episode_result, update)
                print('best model checkpoint saved!')


            if self.config['save_model'] and update % self.config['save_model_frequency'] == 0:
                if not os.path.exists(os.path.join('../../checkpoints', self.config['log_name'])):
                    os.makedirs(os.path.join('../../checkpoints', self.config['log_name']))
                self._save_model(os.path.join('../../checkpoints', self.config['log_name'], self.timestamp.strip('/') + '.pt'), episode_result, update)
                print('checkpoint saved!')



            


        # Save the trained model at the end of the training


    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        
        # Init episodic memory buffer using each workers' current episodic memory
        self.buffer.memories = [self.memory[w] for w in range(self.num_workers)]
        for w in range(self.num_workers):
            self.buffer.memory_index[w] = w

        # Sample actions from the model and collect experiences for optimization
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # print(sliced_memory)
                # print('hhhhhhhhhhhhhhhhhhhhhhhhhhh')
                # print(self.obs, self.obs.shape)
                # print(type(self.obs))
                # print(torch.tensor(self.obs))
                # print('hhhhhhhhhhhhhhhhhhhhhhhhhhh')



                # Store the initial observations inside the buffer
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                #print(self.buffer.obs)
                # print(self.buffer.memory_indices)
                # print(self.worker_current_episode_step)
                # print(len(self.memory_indices))
                # print(self.memory_indices[self.worker_current_episode_step])
                # Store mask and memory indices inside the buffer
                self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
                # print(len(self.memory_indices), self.worker_current_episode_step)
                self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])
                # Forward the model to retrieve the policy, the states' value and the new memory item

                
                # print('DUBUUUUUUG')
                # print(torch.tensor(self.obs), torch.tensor(self.obs).shape)
                policy, value, memory = self.model(torch.tensor(self.obs), sliced_memory, self.buffer.memory_mask[:, t],
                                                   self.buffer.memory_indices[:,t])
                
                # Add new memory item to the episodic memory
                self.memory[self.worker_ids, self.worker_current_episode_step] = memory

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
                self.buffer.values[:, t] = value

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info: # i.e. done
                    # Reset the worker's current timestep
                    self.worker_current_episode_step[w] = 0
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset the agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Break the reference to the worker's memory
                    mem_index = self.buffer.memory_index[w, t]
                    self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()
                    # Reset episodic memory
                    self.memory[w] = torch.zeros((self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
                    if t < self.config["worker_steps"] - 1:
                        # Store memory inside the buffer
                        self.buffer.memories.append(self.memory[w])
                        # Store the reference of to the current episodic memory inside the buffer
                        self.buffer.memory_index[w, t + 1:] = len(self.buffer.memories) - 1
                else:
                    # Increment worker timestep
                    self.worker_current_episode_step[w] +=1
                # Store latest observations
                self.obs[w] = obs
                            
        # Compute the last value of the current observation and memory window to compute GAE
        last_value = self.get_last_value()
        # Compute advantages
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def get_last_value(self):
        """Returns:
                {torch.tensor} -- Last value of the current observation and memory window to compute GAE"""
        start = torch.clip(self.worker_current_episode_step - self.memory_length, 0)
        end = torch.clip(self.worker_current_episode_step, self.memory_length)
        indices = torch.stack([torch.arange(start[b],end[b]) for b in range(self.num_workers)]).long()
        sliced_memory = batched_index_select(self.memory, 1, indices) # Retrieve the memory window from the entire episode
        _, last_value, _ = self.model(torch.tensor(self.obs),
                                        sliced_memory, self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                        self.buffer.memory_indices[:,-1])
        return last_value

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {tuple} -- Training and gradient statistics of one training epoch"""
        train_info, grad_info = [], {}
        for _ in range(self.config["epochs"]):
            mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                # print('minibatch_traning!')
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
                for key, value in self.model.get_grad_norm().items():
                    grad_info.setdefault(key, []).append(value)
        return train_info, grad_info

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Select episodic memory windows
        memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
        
        # Forward model
        # print(f'batchde obs shape: {samples["obs"].shape}')
        # print(f'batchde memory_mask shape: {samples["memory_mask"].shape}')
        # print(f'batchde memory shape: {memory.shape}')

        # print(samples["obs"].shape)
        # print(f'minibatch traning obs: {samples["obs"]}')
        # print(f'minibatch traning memory: {memory}')
        # print(f'minibatch traning memory_mask: {samples["memory_mask"]}')
        # print(f'minibatch traning memory_indices: {samples["memory_indices"]}')

        # print(samples["obs"], samples["obs"].shape)

        policy, value, _ = self.model(samples["obs"], memory, samples["memory_mask"], samples["memory_indices"])

        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        # Monitor additional training stats
        approx_kl = (ratio - 1.0) - log_ratio # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > clip_range).float().mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.mean().cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]



    def init_transformer_memory(self, trxl_conf, max_episode_steps, device):
        """Returns initial tensors for the episodic memory of the transformer.

        Arguments:
            trxl_conf {dict} -- Transformer configuration dictionary
            max_episode_steps {int} -- Maximum number of steps per episode
            device {torch.device} -- Target device for the tensors

        Returns:
            memory {torch.Tensor}, memory_mask {torch.Tensor}, memory_indices {torch.Tensor} -- Initial episodic memory, episodic memory mask, and sliding memory window indices
        """
        # Episodic memory mask used in attention
        memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
        # Episdic memory tensor
        memory = torch.zeros((1, max_episode_steps, trxl_conf["num_blocks"], trxl_conf["embed_dim"])).to(device)
        # Setup sliding memory window indices
        repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0), trxl_conf["memory_length"] - 1, dim = 0).long()
        memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in range(max_episode_steps - trxl_conf["memory_length"] + 1)]).long()
        memory_indices = torch.cat((repetitions, memory_indices))
        return memory, memory_mask, memory_indices

    def _evaluate(self):

        self.model.eval()

        total_reward = 0
        num_successes = 0
        total_steps = 0
        n_episode = self.config['eval_episodes']

        for i in range(n_episode):

            done = False
            memory, memory_mask, memory_indices = self.init_transformer_memory(self.config["transformer"], self.eval_env.max_episode_steps, self.device)
            memory_length = self.config["transformer"]["memory_length"]
            eval_seeds = self.config.get("eval_seeds", None)
            t = 0
            ep_reward = 0

            if eval_seeds is not None:
                obs = self.eval_env.reset(eval_seeds[i])    
            else:
                obs = self.eval_env.reset()




            while not done:
                # Prepare observation and memory
                obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=self.device)
                in_memory = memory[0, memory_indices[t].unsqueeze(0)]
                t_ = max(0, min(t, memory_length - 1))
                mask = memory_mask[t_].unsqueeze(0)
                indices = memory_indices[t].unsqueeze(0)
                # Forward model
                policy, value, new_memory = self.model(obs, in_memory, mask, indices)
                memory[:, t] = new_memory
                # Sample action
                action = []
                for action_branch in policy:
                    action.append(action_branch.sample().item())
                # Step environemnt
                # print(f'action: {action}')
                obs, reward, done, info = self.eval_env.step(action)
                # print(f'Action :{action}, obs: {obs.shape}, reward: {reward}, terminated: {done}, info: {info}')

                ep_reward += reward
                t += 1

            if info.get("is_success"):
                num_successes += 1
            total_reward += ep_reward
            total_steps += t
            #self.eval_env.reset()

        self.model.train()
        return (
            num_successes / n_episode,
            total_reward / n_episode,
            total_steps / n_episode,
        )

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """

        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("training/" + key, episode_result[key], update)

        self.writer.add_scalar("training/losses/Loss", training_stats[2], update)
        self.writer.add_scalar("training/losses/Policy_loss", training_stats[0], update)
        self.writer.add_scalar("training/losses/Calue_loss", training_stats[1], update)
        self.writer.add_scalar("training/losses/Entropy", training_stats[3], update)
        self.writer.add_scalar("training/Value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("training/Advantage_mean", torch.mean(self.buffer.advantages), update)
        self.writer.add_scalar("training/other/Clip_fraction", training_stats[4], update)
        self.writer.add_scalar("training/other/KL", training_stats[5], update)
        
    def _write_gradient_summary(self, update, grad_info):
        """Adds gradient statistics to the tensorboard event file.

        Arguments:
            update {int} -- Current PPO Update
            grad_info {dict} -- Gradient statistics
        """
        for key, value in grad_info.items():
            self.writer.add_scalar("training/gradients/" + key, np.mean(value), update)

    def _write_evaluation_summary(self, update, evaluation_stats) -> None:

        self.writer.add_scalar("evaluation/Return", evaluation_stats[1], update)
        self.writer.add_scalar("evaluation/Success_Rate", evaluation_stats[0], update)
        self.writer.add_scalar("evaluation/Episode_Length", evaluation_stats[2], update)

    def _save_model(self, checkpoint_path, episode_result, update_step) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        # if not os.path.exists("./models"):
        #     os.makedirs("./models")
        # self.model.cpu()
        # pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        # print("Model saved to " + "./models/" + self.run_id + ".nn")

        torch.save(
            {
                "update_step": update_step,
                #
                "reward_mean" : episode_result["reward_mean"],
                "reward_std" : episode_result["reward_std"],
                "length_mean": episode_result["length_mean"],
                "length_std": episode_result["length_std"], 
                #
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # RNG states
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else torch.get_rng_state(),
            },
            checkpoint_path
        )
        




    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)
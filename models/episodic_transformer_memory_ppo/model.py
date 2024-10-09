import numpy as np
import torch

from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable

import gymnasium
import gym

from transformer import Transformer

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_length):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box, discrete} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.memory_layer_size = config["transformer"]["embed_dim"]
        self.observation_space = observation_space

        #self.observation_space_shape = observation_space.shape

        self.max_episode_length = max_episode_length

        # Observation encoder
        if self.observation_space.obs_type == 'image':
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(self.observation_space.obs_shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(self.observation_space.obs_shape)
            in_features_next_layer = self.conv_out_size
        elif self.observation_space.obs_type == 'vector':
            # Case: vector observation is available
            in_features_next_layer = self.observation_space.obs_shape[0] 

        elif self.observation_space.obs_type == 'discrete':
            # Case: discrete observation is available
            # TO-DO: is that good embedding_dim?
            in_features_next_layer = self.observation_space.obs_shape[0] #self.memory_layer_size

            # TO-DO: not correct naming fix?
            # nn.Sequential(nn.Embedding(3, 10), nn.Flatten(0, 1))
            # self.lin_hidden = nn.Sequential(
            #     nn.Embedding(self.observation_space.n, config["transformer"]['embed_per_obs_dim']),
            #     nn.Flatten(start_dim=-2),
            #     nn.Linear(config["transformer"]['embed_per_obs_dim'] * 1, self.memory_layer_size)
            #     )

        elif self.observation_space.obs_type == 'multidiscrete':
            in_features_next_layer = self.observation_space.obs_shape[0]#self.memory_layer_size
            # self.observation_space.nvec.max() + 1 or without + 1
            # self.lin_hidden = nn.Sequential(
            #     nn.Embedding(max(self.observation_space.nvec) + 1, config["transformer"]['embed_per_obs_dim']),
            #     nn.Flatten(start_dim=-2),
            #     nn.Linear(config["transformer"]['embed_per_obs_dim'] * self.observation_space.obs_shape[0], self.memory_layer_size)
            #     )

        # elif self.observation_space.obs_type == 'iterable_discrete':
        #     # Case: list/tuple etc. of discrete observation is available
        #     # TO-DO: is that good embedding_dim?
        #     num_embeddings = sum([obs_space.n for obs_space in observation_space.spaces])
        #     in_features_next_layer = self.memory_layer_size

        #     # TO-DO: incorrect naming fix? - nn.Embedding is not lin_hidden
        #     self.lin_hidden = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = self.memory_layer_size)
        else:
            # Case: list/tuple etc. of image/vector observation is available
            raise NotImplementedError

        
        # Hidden layer

        self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)

        # if self.observation_space.obs_type in ['image', 'vector']:
        #     self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)
        # elif self.observation_space.obs_type in ['discrete', 'iterable_discrete', 'multidiscrete']:
        #     # defined abouve 
        #     pass
        # else:
        #     raise NotImplementedError

        # TO-DO: a
        if isinstance(self.lin_hidden, nn.Sequential):
            for module in self.lin_hidden: # is that correct?
                try: 
                    nn.init.orthogonal_(module.weight, np.sqrt(2))
                except Exception as e:
                    print(f'[WARNING]: {e}')
        else:
            nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Transformer Blocks
        self.transformer = Transformer(config["transformer"], self.memory_layer_size, self.max_episode_length)

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)
            
        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, memory:torch.tensor, memory_mask:torch.tensor, memory_indices:torch.tensor):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            memory {torch.tensor} -- Episodic memory window
            memory_mask {torch.tensor} -- Mask to prevent the model from attending to the padding
            memory_indices {torch.tensor} -- Indices to select the positional encoding that matches the memory window

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value function: Value
        """
        # Set observation as input to the model
        # if self.observation_space.obs_type in ['discrete', 'iterable_discrete', 'multidiscrete']:
        #     h = obs.int()
        # else:
        #     h = obs
        h = obs
        # Forward observation encoder
        if self.observation_space.obs_type == 'image':
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # # Feed hidden layer
        
        # print(h.size())
        # print(self.observation_space.obs_type)
        # print(f'emb_after_hidden:{self.lin_hidden(h).shape}')
        # print(f'mem:{memory, memory.shape}')
        # print(f'mem_mask:{memory_mask, memory_mask.shape}')
        # print(f'mem_ind:{memory_indices,memory_indices.shape}')


        # print(f'h init shape: {h.shape}')
        # print(f'DEBUUUUUUG_1{h.shape}')
        # print(f"DEGUUUUUUUUUB {h}")
        h = F.relu(self.lin_hidden(h))
        # print(f'DEBUUUUUUG_2{h.shape}')
        # print(f'h after emb shape: {h.shape}')
        # print(f'model forward hidden shape {h.shape}')
        # print(f'memory shape {memory.shape}')
        # print(f'memory mask shape {memory_mask.shape}')
        
        # Forward transformer blocks
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)
        # print(f'DEBUUUUUUG_3{h.shape}')

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]
        
        return pi, value, memory

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
        
    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by layer name
        """
        grads = {}
        if self.observation_space.obs_type == 'image':
            grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3)  
            
        grads["linear_layer"] = self._calc_grad_norm(self.lin_hidden)
        
        transfomer_blocks = self.transformer.transformer_blocks
        for i, block in enumerate(transfomer_blocks):
            grads["transformer_block_" + str(i)] = self._calc_grad_norm(block)
        
        for i, head in enumerate(self.policy_branches):
            grads["policy_head_" + str(i)] = self._calc_grad_norm(head)
        
        grads["lin_policy"] = self._calc_grad_norm(self.lin_policy)
        grads["value"] = self._calc_grad_norm(self.lin_value, self.value)
        grads["model"] = self._calc_grad_norm(self, self.value)
          
        return grads
    
    def _calc_grad_norm(self, *modules):
        """Computes the norm of the gradients of the given modules.

        Arguments:
            modules {list} -- List of modules to compute the norm of the gradients of.

        Returns:
            {float} -- Norm of the gradients of the given modules. 
        """
        grads = []
        for module in modules:
            for name, parameter in module.named_parameters():
                grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None
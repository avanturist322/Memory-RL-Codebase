import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from dtqn.networks.representations import EmbeddingRepresentation
from dtqn.networks.gates import GRUGate, ResGate
from dtqn.networks.transformer import TransformerLayer, TransformerIdentityLayer # do not forget to remove 
from dtqn.networks.mamba import Block
from transformers.modeling_utils import PreTrainedModel  #, Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


# This function taken from the torch transformer tutorial
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def sinusoidal_pos(
    context_len: int,
    embed_dim: int,
):
    position = torch.arange(context_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    pos_encoding = torch.zeros(1, context_len, embed_dim)
    pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
    pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
    return pos_encoding


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])  #, scale=True
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class DMQN(nn.Module):
    """Deep Mamba Q-Network for partially observable reinforcement learning.

    Args:
        obs_dim: The length of the observation vector.
        num_actions: The number of possible environments actions.
        embed_per_obs_dim: Used for discrete observation space. Length of the embed for each
            element in the observation dimension.
        n_inner: The dimensionality of the network. Referred to as d_k by the
            original transformer.
        num_heads: The number of heads to use in the MultiHeadAttention.
        history_len: The maximum number of observations to take in.
        drop_p: Dropout percentage. Default: `0.0`
        gate: Which layer to use after the attention and feedforward submodules (choices: `res`
            or `gru`). Default: `res`
        identity: Whether or not to use identity map reordering. Default: `False`
        pos: The kind of position encodings to use. `0` uses no position encodings, `1` uses
            learned position encodings, and `sin` uses sinusoidal encodings. Default: `1`
        discrete: Whether or not the environment has discrete observations. Default: `False`
        vocab_sizes: If discrete env only. Represents the number of observations in the
            environment. If the environment has multiple obs dims with different number
            of observations in each dim, this can be supplied as a vector. Default: `None`
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int, #embed_per_obs_dim
        n_inner: int, # inner_embed_size
        num_heads: int,
        num_layers: int,
        history_len: int,
        drop_p: float = 0.0, #dropout
        gate: str = "res",
        identity: bool = False,
        pos: Union[str, int] = 1,
        discrete: bool = False,
        vocab_sizes: Optional[Union[np.ndarray, int]] = None,
        model_type = 'mamba', # 'mamba2' 'mamba-lite'
        # layer_norm_epsilon = , 
        
    ):
        super().__init__()

        n_embd = n_inner

        self.obs_dim = obs_dim
        self.discrete = discrete
        # Input Embedding
        if discrete:
            self.obs_embedding = EmbeddingRepresentation.make_discrete_representation(
                vocab_sizes=vocab_sizes,
                obs_dim=obs_dim,
                embed_per_obs_dim=embed_per_obs_dim,
                outer_embed_size=n_inner,
            )
        else:
            self.obs_embedding = EmbeddingRepresentation.make_continuous_representation(
                obs_dim=obs_dim, outer_embed_size=n_inner
            )

        # If pos is 0, the positional embeddings are just 0s. Otherwise, they become learnables that initialise as 0s
        try:
            int(pos)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, history_len, n_inner),
                requires_grad=False if pos == 0 else True,
            )
        except ValueError:
            if pos == "sin":
                self.position_embedding = nn.Parameter(
                    sinusoidal_pos(context_len=history_len, embed_dim=n_inner),
                    requires_grad=False,
                )
            else:
                raise AssertionError(f"pos must be either int or sin but was {pos}")
        self.drop_p = nn.Dropout(drop_p)

        if gate == "gru":
            attn_gate = GRUGate(embed_size=n_inner)
            mlp_gate = GRUGate(embed_size=n_inner)
        elif gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        if identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        # self.transformer_layers = nn.Sequential(
        #     *[
        #         transformer_block(
        #             num_heads,
        #             n_inner,
        #             history_len,
        #             drop_p,
        #             attn_gate,
        #             mlp_gate,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )

        self.transformer_layers = nn.Sequential(
            *[
                Block(
                    model_type,
                    n_embd,
                    n_inner, 
                    # layer_norm_epsilon, 
                    drop_p,
                )
                for _ in range(num_layers)
            ]
        )

        

        self.layernorm = nn.LayerNorm(n_inner)
        self.ffn = nn.Sequential(
            nn.Linear(n_inner, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, num_actions),
        )

        self.history_len = history_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.MultiheadAttention)):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.in_proj_bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obss: torch.Tensor) -> torch.Tensor:
        batch, history_len, obs_dim = obss.size()
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."
        assert (
            obs_dim == self.obs_dim
        ), f"Obs dim is incorrect. Expected {self.obs_dim} got {obs_dim}"

        token_embeddings = self.obs_embedding(obss)
        # batch_size x hist_len x obs_dim
        x = self.drop_p(token_embeddings + self.position_embedding[:, :history_len, :])
        # Send through transformer
        x = self.transformer_layers(x)
        # Norm and run through a linear layer to get to action space
        x = self.layernorm(x)
        return self.ffn(x)

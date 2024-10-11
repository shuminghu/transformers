# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
# from transformers.models.mllama.modeling_mllama import MllamaPrecomputedAspectRatioEmbedding, MllamaVisionEncoder
from ...utils import TensorType, logging


import torch
from torch import nn
import torch.nn.functional as F
from ...activations import ACT2FN



logger = logging.get_logger(__name__)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, input: torch.Tensor):
        x = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class MPPerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = LayerNorm(dim)
        self.norm_latents = LayerNorm(dim)

        self.to_q = nn.Linear(
            dim, inner_dim, bias=False,
        )
        self.to_kv = nn.Linear(
            dim, inner_dim * 2, bias=False,
        )
        self.to_out = nn.Linear(
            inner_dim, dim, bias=False,
        )

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        bs, slen, _ = q.shape

        q = q.view(bs, slen, h, self.dim_head)
        k = k.view(bs, k.shape[1], h, self.dim_head)
        v = v.view(bs, v.shape[1], h, self.dim_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).contiguous()
        out = out.view(bs, slen, -1)

        out = self.to_out(out)

        return out


class MPFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        act_layer: str = "gelu",
    ):
        super().__init__()
        # layers
        self.c_fc = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )
        self.c_proj = nn.Linear(
            hidden_dim,
            dim,
            bias=bias,
        )
        self.non_linearity = ACT2FN[act_layer]
        self.dropout = dropout

    def forward(self, x):
        hidden = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.c_proj(hidden)


class MPTemporalEmbedding(nn.Module):
    def __init__(
        self,
        num_frames: int,
        dim: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.dim = dim
        self.embedding = nn.Parameter(
            torch.randn(num_frames, 1, 1, self.dim) / math.sqrt(self.dim)
        )
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # load the weights from the checkpoint
        embed = state_dict.get(prefix + "embedding")
        if embed is not None:
            # reshape the weights to the correct shape
            nf_old, _, _, d = embed.shape
            embed = embed.permute(1, 2, 3, 0)
            # interpolate needs 3D for temporal linear interpolation
            embed_new = F.interpolate(
                embed.squeeze(0), size=self.num_frames, mode="linear"
            ).unsqueeze(0)

            logger.info(
                f"Resizing temporal embedding from {nf_old} to {self.num_frames}"
            )
            # reshape the weights to the correct shape
            embed_new = embed_new.permute(3, 0, 1, 2)
            # assign the weights to the module
            state_dict[prefix + "embedding"] = embed_new

    def forward(self, x: torch.Tensor):
        B, F, C, T, D = x.shape

        assert (
            F <= self.num_frames
        ), "Input frames F has to be smaller or equal than self.num_frames"

        if self.gated:
            x = x + self.gate.tanh() * self.embedding[:F]
        else:
            x = x + self.embedding[:F]
        return x


class MPPerceiverResamplerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        ff_mult: float,
        act_layer: str = "gelu",
    ):
        super().__init__()
        self.attn = MPPerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.norm = LayerNorm(dim)
        self.ff = MPFeedForward(
            dim,
            int(dim * ff_mult),
            bias=False,
            dropout=0.0,
            act_layer=act_layer,
        )

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
    ):
        latents = self.attn(x, latents) + latents
        latents = self.ff(self.norm(latents)) + latents
        return latents


class MPPerceiverResampler(nn.Module):
    def __init__(
        self,
        config: MllamaVisionConfig,
    ):
        super().__init__()

        num_latents = config.perceiver_num_latents
        num_layers = config.perceiver_num_layers
        dim_head = config.perceiver_dim_head
        heads = config.perceiver_heads
        ff_mult = config.perceiver_ff_mult
        dim = config.perceiver_dim

        add_post_tile_pos_embed = (
            config.perceiver_add_post_tile_pos_embed
        )
        num_post_global_attention = (
            config.perceiver_num_post_global_attention
        )

        self.num_frames = config.num_frames
        self.frames_per_group = config.frames_per_group

        self.input_dim = config.perceiver_input_dim
        if dim is not None:
            # we support having a different dim for the perceiver compared to the
            # input dim. in this situation, we need to apply projections to go from
            # input_dim to dim
            assert dim != self.input_dim
            self.dim = dim
            self.pre_proj = nn.Linear(
                self.input_dim, self.dim, bias=True,
            )
            self.post_proj = nn.Linear(
                self.dim, self.input_dim, bias=True,
            )
        else:
            self.dim = config.input_dim
            self.pre_proj = None
            self.post_proj = None

        self.latents = nn.Parameter(
            torch.randn(num_latents, self.dim) / math.sqrt(self.dim)
        )
        self.frame_embs = MPTemporalEmbedding(
            num_frames=self.num_frames,
            dim=self.dim,
            gated=False,
        )

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                MPPerceiverResamplerBlock(
                    dim=self.dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                )
            )

        self.post_tile_pos_embed = None
        if add_post_tile_pos_embed:
            # reference: https://www.internalfb.com/code/fbsource/[95374dc8728e]/fbcode/gen_ai/mllm/inference/llama3/model/video.py?lines=282
            # self.post_tile_pos_embed = MllamaPrecomputedAspectRatioEmbedding(config)
            raise NotImplementedError("Post tile position embedding is not implemented")
        self.post_global_attention = None
        if num_post_global_attention is not None:
            perceiver_global_attention_config = config.copy()
            perceiver_global_attention_config.hidden_size = dim_head * heads
            perceiver_global_attention_config.attention_heads = heads
            perceiver_global_attention_config.intermediate_size = ff_mult * dim_head * heads
            # reference: https://www.internalfb.com/code/fbsource/[95374dc8728e]/fbcode/gen_ai/mllm/inference/llama3/model/video.py?lines=288
            # self.post_global_attention = MllamaVisionEncoder(perceiver_global_attention_config, num_layers=num_post_global_attention, is_gated=False)
            raise NotImplementedError("Post global attention is not implemented")
        
        self.norm = LayerNorm(self.dim)


    def forward(self, x, ar):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, F, C, T, D)
                b - batch size
                F - number of frames
                C - number of chunks
                T - number of tokens
                D - feature dimension
            ar (torch.Tensor): aspect ratio
        Returns:
            shape (b, 1, C, T1, D) where T1 is self.num_latents
        """
        feature_dim = self.dim
        b, F, C, T, D = x.shape
        T1, _ = self.latents.shape

        assert self.input_dim == D, f"{self.input_dim} != {D}"
        assert (
            # For the case of 1-frame inference for images,
            # or when fewer frames are used during SFT etc.
            F < self.frames_per_group
            or
            # For more frames being used, it must be a multiple
            # of the frames_per_group to process each group separately.
            F % self.frames_per_group == 0
        ), (
            f"Input frames {F} has to be <= {self.frames_per_group} "
            "or divisible by it."
        )

        # pre-projection
        if self.pre_proj is not None:
            x = self.pre_proj(x)
            b, F, C, T, D = x.shape

        # Add the temporal embeddings first. This ensures even after
        # the grouping, there is some global position embedding in
        # the features (hence better than re-using the same position
        # embeddings for each group separately instead).
        x = self.frame_embs(x)

        # Move the group dimension to the batch size and update
        if F < self.frames_per_group:
            num_groups = 1
            effective_frames_per_group = F
        else:
            num_groups = F // self.frames_per_group
            effective_frames_per_group = self.frames_per_group
        x = x.view(b, num_groups, effective_frames_per_group, C, T, D)
        x = torch.flatten(x, 0, 1)
        b = b * num_groups
        F = effective_frames_per_group

        # perceiver blocks
        x = x.permute(0, 2, 1, 3, 4)
        # perceiver across the temporal dimension of each chunk separately
        x = x.reshape(b * C, F * T, feature_dim)
        latents = self.latents.unsqueeze(0).expand(b * C, -1, -1)
        for layer in self.layers:
            latents = layer(x, latents)
        latents = latents.reshape(b, C, 1, T1, feature_dim)
        latents = latents.permute(0, 2, 1, 3, 4)

        # latent position embedding
        if self.post_tile_pos_embed is not None:
            latents = latents.reshape(b, C, T1, feature_dim)
            # all ars are the same because its frames of video
            # hence we take for first frame ar[:, 0, :]
            latents = self.post_tile_pos_embed(latents, ar[:, 0, :])
            latents = latents.reshape(b, 1, C, T1, feature_dim)

        # attention over latents of all chunks
        if self.post_global_attention is not None:
            latents = latents.reshape(b, C * T1, feature_dim)
            latents = self.post_global_attention(latents)
            latents = latents.reshape(b, 1, C, T1, feature_dim)

        # Reshape back to get the group dimension
        latents = torch.unflatten(latents, 0, (-1, num_groups))
        latents = torch.flatten(latents, 1, 2)

        latents = self.norm(latents)

        # post-projection
        if self.post_proj is not None:
            latents = self.post_proj(latents)

        return latents

import math
from abc import abstractmethod
from typing import Tuple
import os
import torch
from torch import Tensor, nn

from .attention import (
    CrossAttention,
    FeedForward,
    MultiHeadAttention,
)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        drop: float = 0.0,
        mlp_ratio: int = 4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = MultiHeadAttention(kv_dim=dim, q_dim=dim, num_heads=num_heads)
        self._num_heads = num_heads

        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.mlp = FeedForward(dim, widening_factor=mlp_ratio, dropout=drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

    @abstractmethod
    def pre_attn(
        self, scene_tensor: Tensor, context: Tensor, valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def post_attn(self, scene_tensor: Tensor, batch_size: int) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, c: Tensor, valid_mask: Tensor) -> Tensor:
        B, NA, NT, ND = x.size()

        x, attn_mask, c = self.pre_attn(x, c, valid_mask)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        queries = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(
            queries,
            queries,
            attn_mask.unsqueeze(1).repeat(1, self._num_heads, 1, 1).bool(),
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.post_attn(x, B)
        return x


class TemporalBlock(AdaLNBlock):
    def pre_attn(
        self, scene_tensor: Tensor, context: Tensor, valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        scene_tensor: tensor of shape (B, NA, NT, ND)
        valid_mask: tensor of shape (B, NA, NT)
        """
        B, NA, NT, ND = scene_tensor.size()
        scene_tensor = scene_tensor.reshape(B * NA, NT, ND)
        context = context.reshape(B * NA, NT, ND)

        # reshape the valid mask to be (B * NA, NT, 1)
        valid_mask = valid_mask.unsqueeze(-1).reshape(B * NA, NT, 1)
        # take the outer product of the valid mask with itself to get the attention mask
        attn_mask = valid_mask @ valid_mask.transpose(1, 2)

        return scene_tensor, attn_mask, context

    def post_attn(self, scene_tensor: Tensor, batch_size: int) -> Tensor:
        return scene_tensor.reshape(
            batch_size, -1, scene_tensor.size(-2), scene_tensor.size(-1)
        )


class SpatialBlock(AdaLNBlock):
    def pre_attn(
        self, scene_tensor: Tensor, context: Tensor, valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        scene_tensor: tensor of shape (B, NA, NT, ND)
        valid_mask: tensor of shape (B, NA, NT)
        """
        B, NA, NT, ND = scene_tensor.size()
        scene_tensor = scene_tensor.permute(0, 2, 1, 3).reshape(B * NT, NA, ND)
        context = context.permute(0, 2, 1, 3).reshape(B * NT, NA, ND)

        # reshape the valid mask to be (B * NT, NA, 1)
        valid_mask = valid_mask.permute(0, 2, 1).unsqueeze(-1).reshape(B * NT, NA, 1)
        # take the outer product of the valid mask with itself to get the attention mask
        attn_mask = valid_mask @ valid_mask.transpose(1, 2)

        # attn_mask = torch.ones(B * NT, NA, NA, device=scene_tensor.device)
        return scene_tensor, attn_mask, context

    def post_attn(self, scene_tensor: Tensor, batch_size: int) -> Tensor:
        return scene_tensor.reshape(
            batch_size, -1, scene_tensor.size(-2), scene_tensor.size(-1)
        ).permute(0, 2, 1, 3)


class CombinedAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.temporal_attention = TemporalBlock(hidden_dim)
        self.spatial_attention = SpatialBlock(hidden_dim)

    def forward(
        self, scene_tensor: Tensor, context: Tensor, valid_mask: Tensor
    ) -> Tensor:
        scene_tensor = self.temporal_attention(scene_tensor, context, valid_mask)
        scene_tensor = self.spatial_attention(scene_tensor, context, valid_mask)
        return scene_tensor


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLP(nn.Module):
    """Transformer Feed-Forward network."""

    def __init__(self, input_dim: int, output_dim: int, widening_factor: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * widening_factor),
            nn.GELU(),
            nn.Linear(output_dim * widening_factor, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class UNet(nn.Module):
    """Based on the Nexus paper. Perciever I/O structure"""

    def __init__(
        self,
        nblocks: int = 4,
        hidden_dim: int = 256,
        scene_tensor_dim: int = 16,
        cross_attention_num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scene_tensor_dim = scene_tensor_dim
        self._num_heads = cross_attention_num_heads
        self.cross_attention = CrossAttention(
            kv_dim=hidden_dim,
            q_dim=hidden_dim,
            qk_out_dim=hidden_dim,
            v_out_dim=hidden_dim,
            num_heads=self._num_heads,
        )
        self.input_proj = nn.Linear(
            3 * scene_tensor_dim, hidden_dim, bias=False
        )
        self.input_proj_global = nn.Linear(
            scene_tensor_dim, hidden_dim, bias=False
        )
        # self.input_proj = nn.Linear(
        #     scene_tensor_dim, hidden_dim, bias=False
        # )
        self.physical_timestep_embeddor = TimestepEmbedder(hidden_dim)
        self.diffusion_timestep_embeddor = TimestepEmbedder(hidden_dim)

        self.backbone = nn.Sequential(
            *[CombinedAttention(hidden_dim) for _ in range(nblocks)]
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, scene_tensor_dim)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # init first layer
        w = self.input_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_proj.bias, 0)

        w = self.input_proj_global.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_proj_global.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.physical_timestep_embeddor.mlp[0].weight, std=0.02)
        nn.init.normal_(self.physical_timestep_embeddor.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.diffusion_timestep_embeddor.mlp[0].weight, std=0.02)
        nn.init.normal_(self.diffusion_timestep_embeddor.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.backbone:
            nn.init.constant_(block.temporal_attention.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.temporal_attention.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.spatial_attention.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.spatial_attention.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        local_context: torch.Tensor,
        diffused_scene_tensor: torch.Tensor,
        valid_mask: torch.Tensor,
        diffusion_times: torch.Tensor,
        global_context: torch.Tensor,
    ):
        """
        Args:
            scene_tensor: tensor of shape (B, NA, NT, ND)
            valid_mask: tensor of shape (B, NA, NT)
            diffusion_times: tensor of shape (B, NA, NT)
            valid_mask: tensor of shape (B, NA, NT)

        """
        # x is the queries, and global context is the keys and values
        B, NA, NT, ND = local_context.size()

        queries = self.input_proj.forward(
            torch.cat([local_context, diffused_scene_tensor], dim=-1)
        )
        # try to remove local context
        # queries = self.input_proj.forward(diffused_scene_tensor)

        # positional encode the scene tensor, both diffusion timestep and physical timestep
        diffusion_time_embed = self.diffusion_timestep_embeddor(
            diffusion_times.view(-1)
        ).view(B, NA, NT, -1)
        physical_time_embed = (
            self.physical_timestep_embeddor(
                (torch.arange(NT, device=local_context.device) / NT).view(-1),
            )
            .view(1, 1, NT, -1)
            .expand(B, NA, NT, -1)
        )

        # add the positional embeddings to the queries
        if os.environ.get("NO_PHYPE", False):
            queries += diffusion_time_embed
        else:
            queries += (diffusion_time_embed + physical_time_embed)    

        global_context = self.input_proj_global.forward(global_context)

        # note that we dont have to use the valid mask here as
        fused_context = self.cross_attention(
            inputs_kv=global_context,
            inputs_q=queries.view(B, NA * NT, self.hidden_dim),
        ).view(queries.size()) * valid_mask.unsqueeze(-1)

        for i, block in enumerate(self.backbone):
            queries = block(
                queries,
                fused_context,
                valid_mask,
            )

        shift, scale = self.adaLN_modulation(fused_context).chunk(2, dim=-1)
        queries = modulate(self.norm_final(queries), shift, scale)
        denoised_scene_tensor = self.output_proj(queries)

        return denoised_scene_tensor
from functools import partial
from typing import Optional, Literal
import torch
from torch import nn
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from .embeddings import Timesteps, TimestepEmbedding
from .attention import SpatialAttentionBlock, TemporalAttentionBlock
from .resnet import ResnetBlock, Downsample, Upsample
from .utils import default
import math
from perceiver.model.core.modules import CrossAttention

class NoiseLevelSequential(nn.Sequential):
    """
    Sequential module that passes the noise level to each module in the sequence if it accepts it.
    """

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor):
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, noise_level)
            else:
                x = module(x)
        return x


class Unet3D(nn.Module):
    # Special thanks to lucidrains for the implementation of the base Diffusion model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        external_cond_dim: Optional[int] = None,
        channels=3,
        resnet_block_groups=8,
        dim_mults=[1, 2, 4, 8],
        attn_resolutions=[1, 2, 4, 8],
        attn_dim_head=32,
        attn_heads=4,
        use_linear_attn=True,
        use_init_temporal_attn=True,
        init_kernel_size=7,
        is_causal=True,
        time_emb_type: Literal["sinusoidal", "rotary"] = "rotary",
    ):
        super().__init__()
        self.channels = channels
        # if external_cond_dim:
        #     raise NotImplementedError("External conditioning not yet implemented")
        self.external_cond_dim = external_cond_dim
        init_dim = default(init_dim, dim)
        out_dim = default(out_dim, channels)
        self.is_causal = is_causal
        dim_mults = list(dim_mults)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        noise_level_emb_dim = dim * 4
        self.noise_level_pos_embedding = nn.Sequential(
            Timesteps(dim, True, 0),
            TimestepEmbedding(in_channels=dim, time_embed_dim=noise_level_emb_dim),
        )

        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head) if time_emb_type == "rotary" else None

        init_padding = init_kernel_size // 2
        init_kernel_size, init_padding = 1, 0
        self.init_conv = nn.Conv3d(
            channels+self.external_cond_dim,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = (
            TemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=is_causal,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else nn.Identity()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        block_klass_noise = partial(ResnetBlock, groups=resnet_block_groups, emb_dim=noise_level_emb_dim)
        spatial_attn_klass = partial(SpatialAttentionBlock, heads=attn_heads, dim_head=attn_dim_head)
        temporal_attn_klass = partial(
            TemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=is_causal,
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1
        use_attn = True

        for idx, (dim_in, dim_out) in enumerate(in_out):
            # is_last = idx == len(in_out) - 1
            is_last = True
            # use_attn = curr_resolution in attn_resolutions

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        NoiseLevelSequential(
                            block_klass_noise(dim_in, dim_out),
                            block_klass_noise(dim_out, dim_out),
                            # spatial_attn_klass(dim_out),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
                                )
                                if use_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1

        self.mid_block = NoiseLevelSequential(
            block_klass_noise(mid_dim, mid_dim),
            # spatial_attn_klass(mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            # is_last = idx == len(in_out) - 1
            is_last = True
            # use_attn = curr_resolution in attn_resolutions

            self.up_blocks.append(
                NoiseLevelSequential(
                    block_klass_noise(dim_out * 2, dim_in),
                    block_klass_noise(dim_in, dim_in),
                    # spatial_attn_klass(dim_in),
                    (spatial_attn_klass(dim_in, use_linear=use_linear_attn and idx > 0) if use_attn else nn.Identity()),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                )
            )

            curr_resolution //= 2 if not is_last else 1

        self.out = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1))

        hidden_dim = 384
        self.cross_attention = CrossAttention(
            num_q_input_channels=512,
            num_kv_input_channels=384,
            num_heads=4,
        )

    def timestep_embedding(self, timesteps, dim, max_period=10000):
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

    def project_global_feature(self, h, cond_feature, noise_levels):

        B, ND, NT, NA, _ = h.shape
        # cond_feature[0]: [B, 169, 384]
        # noise_levels: [f, b, a]
        
        physical_time_embed = (
            self.timestep_embedding(
                (torch.arange(NT, device=h.device) / NT).view(-1),
                ND,
                max_period=1,
            )
            .view(1, NT, 1, ND)
            .expand(B, NT, NA, ND)
        )

        # add the positional embeddings to the queries
        noise_levels = rearrange(noise_levels, "f b a -> (b f a)")
        diffusion_time_embed = self.timestep_embedding(
            noise_levels / 1000, ND, max_period=1
        ).view(B, NT, NA, ND)

        queries = rearrange(h, "B ND NT NA 1 -> B NT NA ND") + (diffusion_time_embed + physical_time_embed)
        # queries = rearrange(h, "B ND NT NA 1 -> B NT NA ND")
        global_context = cond_feature[0]

        valid_mask = repeat(torch.from_numpy(cond_feature[2][:,:NT]).to(h.device), "b t a -> b (t a) ND", ND=ND)

        # note that we dont have to use the valid mask here as
        fused_context = self.cross_attention(
            x_kv=global_context,
            x_q=queries.reshape(B, NT * NA, ND),
        )["last_hidden_state"].to(h.device) * valid_mask

        fused_context = rearrange(fused_context, "B (NT NA) ND -> B ND NT NA 1", NA=NA, NT=NT).to(h.dtype)

        # import pdb; pdb.set_trace()

        return fused_context
    
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        is_causal: Optional[bool] = None,
    ):
        if is_causal is not None and is_causal != self.is_causal:
            raise ValueError("is_causal must be the same as the one used during initialization")
        if len(noise_levels.shape) == 2:
            noise_levels = rearrange(noise_levels, "f b -> b f")
            noise_level_emb = self.noise_level_pos_embedding(noise_levels)
        else:
            noise_levels_t = rearrange(noise_levels, "f b a -> (b a) f")
            noise_level_emb_t = rearrange(self.noise_level_pos_embedding(noise_levels_t), "(b a) f c -> b f a c", b=x.shape[0])
            noise_levels_a = rearrange(noise_levels, "f b a -> (b f) a")
            noise_level_emb_a = rearrange(self.noise_level_pos_embedding(noise_levels_a), "(b f) a c -> b f a c", b=x.shape[0])
            # noise_level_emb = (noise_level_emb_t + noise_level_emb_a)/2
            noise_level_emb = noise_level_emb_t
        # x [B, C, T, A, 1]
        # cond_feature[1]: [T, B, C, A, 1]
        if external_cond[1] is not None:
            assert self.external_cond_dim > 0
            x = torch.cat([x, rearrange(external_cond[1][:x.shape[2]], "T B C A 1 -> B C T A 1")], dim=1)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x)
        h = x.clone()

        hs = []

        for block, downsample in self.down_blocks:
            h = block(h, noise_level_emb)
            hs.append(h)
            h = downsample(h)
        if external_cond[1] is not None:
            h = self.project_global_feature(h, external_cond, noise_levels)
        h = self.mid_block(h, noise_level_emb)

        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, noise_level_emb)

        h = torch.cat([h, x], dim=1)
        return self.out(h)
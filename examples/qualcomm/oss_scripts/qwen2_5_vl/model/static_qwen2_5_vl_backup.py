# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Based on HuggingFace Transformers Qwen2.5-VL implementation

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig


class ModelArgs:
    """
    Model arguments class similar to LLaMA's ModelArgs but for Qwen2.5-VL
    """
    def __init__(
        self,
        vocab_size: int = 152064,
        dim: int = 8192,
        hidden_dim: int = 29568,
        n_layers: int = 80,
        n_heads: int = 64,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 32768,
        max_batch_size: int = 1,
        norm_eps: float = 1e-5,
        rope_freq_base: float = 1000000.0,
        use_kv_cache: bool = True,
        use_scaled_rope: bool = False,
        rope_scale_factor: float = 8.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.norm_eps = norm_eps
        self.rope_freq_base = rope_freq_base
        self.use_kv_cache = use_kv_cache
        self.use_scaled_rope = use_scaled_rope
        self.rope_scale_factor = rope_scale_factor

    @classmethod
    def from_qwen_config(cls, config: Qwen2_5_VLConfig):
        """Convert Qwen2_5_VLConfig to ModelArgs"""
        return cls(
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            max_batch_size=getattr(config, 'max_batch_size', 1),
            norm_eps=config.rms_norm_eps,
            rope_freq_base=config.rope_theta,
            use_kv_cache=getattr(config, 'use_kv_cache', True),
            use_scaled_rope=getattr(config, 'use_scaled_rope', False),
            rope_scale_factor=getattr(config, 'rope_scale_factor', 8.0),
        )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    # The implementation of RoPE in HuggingFace processes query and key with two half instead of interleaved way.
    # The main difference is stride in StrideSlice op. For interleaved way, stride is two which is not friendly for HTP backend.
    # Ref: https://github.com/huggingface/transformers/issues/25199
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # broadcast for batch_prefill mode input x
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, None, :, :]
        freqs_sin = freqs_sin[None, None, :, :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    scale_factor: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device, dtype=freqs.dtype)
    if use_scaled:
        t = t / scale_factor
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, args: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads
        self.num_key_value_groups = args.n_heads // self.n_kv_heads
        self.max_seq_len = args.max_seq_len
        self.output_new_cache_only = output_new_cache_only

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)
        self.scale = float(self.head_dim) ** 0.5

    def prepare_sha(self):
        self.wq_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=True)
                for _ in range(self.n_heads)
            ]
        )
        self.wk_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=True)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wv_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=True)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wo_sha = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_mha = self.forward
        self.forward = self.forward_sha
        
        for i in range(self.n_heads):
            self.wq_sha[i].weight.data.copy_(
                self.wq.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            if self.wq.bias is not None:
                self.wq_sha[i].bias.data.copy_(
                    self.wq.bias[i * self.head_dim : (i + 1) * self.head_dim]
                )
        
        for i in range(self.n_kv_heads):
            self.wk_sha[i].weight.data.copy_(
                self.wk.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            if self.wk.bias is not None:
                self.wk_sha[i].bias.data.copy_(
                    self.wk.bias[i * self.head_dim : (i + 1) * self.head_dim]
                )
                
            self.wv_sha[i].weight.data.copy_(
                self.wv.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            if self.wv.bias is not None:
                self.wv_sha[i].bias.data.copy_(
                    self.wv.bias[i * self.head_dim : (i + 1) * self.head_dim]
                )
        
        self.wo_sha.weight.data.copy_(self.wo.weight[:, :, None, None])

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: Optional[List[torch.Tensor]] = None,
        v_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        # In the HTP backend, the input axis order for the convolution operation is
        # more efficient with [1, 1, seq_len, dim] compared to [1, seq_len, 1, dim].
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)
        
        q = [
            wq_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wq_sha in self.wq_sha
        ]
        k = [
            wk_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wk_sha in self.wk_sha
        ]
        v = [
            wv_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wv_sha in self.wv_sha
        ]
        
        for i in range(len(q)):
            q[i] = apply_rotary_emb_single(q[i], freqs_cos, freqs_sin)
        for i in range(len(k)):
            k[i] = apply_rotary_emb_single(k[i], freqs_cos, freqs_sin).transpose(1, 2)

        output_y = []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[i]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[i]], dim=1))
        # batch_prefill mode
        else:
            kh = k
            vh = v

        for i, _ in enumerate(q):
            cache_idx = i // self.num_key_value_groups
            attn = q[i] @ kh[cache_idx]
            attn = attn / self.scale + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh[cache_idx]
            output_y.append(y)

        y = torch.concat(output_y, dim=-1)
        y = y.reshape(bsz, seq_len, 1, -1)
        y = y.transpose(1, 3)
        y = self.wo_sha(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        if self.output_new_cache_only:
            return y, k, v

        return y, kh, vh

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape

        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb_single(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb_single(k, freqs_cos, freqs_sin).permute(0, 2, 3, 1)

        output_kh, output_vh, output_y = [], [], []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[:, i, :, :]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[:, :, i, :]], dim=1))
            for i in range(self.n_heads):
                cache_idx = i // self.num_key_value_groups

                attn = q[:, :, i, :] @ kh[cache_idx]
                attn = attn / self.scale + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh[cache_idx]

                output_y.append(y)

        # batch_prefill mode
        else:
            kh = k
            vh = v
            for i in range(self.n_heads):
                cache_idx = i // self.num_key_value_groups

                attn = q[:, :, i, :] @ kh[:, cache_idx, :, :]
                attn = attn / self.scale + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh[:, :, cache_idx, :]

                output_y.append(y)

        for i in range(self.n_kv_heads):
            if self.output_new_cache_only:
                output_kh.append(k[:, i, :, -1])
                output_vh.append(v[:, -1, i, :])
            else:
                output_kh.append(k[:, i, :, :])
                output_vh.append(v[:, :, i, :])

        y = torch.concat(output_y, dim=-1)
        y = self.wo(y)

        return y, output_kh, output_vh


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_dim is not None
        self.hidden_dim: int = args.hidden_dim
        self.dim: int = args.dim
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def prepare_feedforward_conv(self):
        self.w1_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)
        self.w2_conv = nn.Conv2d(self.hidden_dim, self.dim, 1, bias=False)
        self.w3_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedforward_conv

        self.w1_conv.weight.data.copy_(self.w1.weight[:, :, None, None])
        self.w2_conv.weight.data.copy_(self.w2.weight[:, :, None, None])
        self.w3_conv.weight.data.copy_(self.w3.weight[:, :, None, None])

        del self.w1
        del self.w2
        del self.w3

    def forward_feedforward_conv(self, x):
        bsz, _, _ = x.size()
        x = torch.reshape(x, (bsz, -1, 1, self.dim))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.w2_conv(F.silu(self.w1_conv(x)) * self.w3_conv(x))
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.dim))
        return x

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = args.dim
        self.attention = Qwen2_5_VLAttention(
            args=args, output_new_cache_only=output_new_cache_only
        )
        self.feed_forward = FeedForward(args)
        self.attention_norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k_cache, v_cache = self.attention(
            hidden_states=self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        h = x + h
        output = h + self.feed_forward(self.ffn_norm(h))
        return output, k_cache, v_cache


class Qwen2_5_VLModel(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.rope_freq_base = args.rope_freq_base
        self.use_kv_cache = args.use_kv_cache
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.output_cache = output_cache

        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer(args, self.output_new_cache_only)
                for _ in range(args.n_layers)
            ]
        )
        self.norm = torch.nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.head_dim,
            args.max_seq_len,
            args.rope_freq_base,
            args.use_scaled_rope,
            args.rope_scale_factor,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def prepare_output_conv(self):
        def forward_output_conv(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.dim))
            x = x.transpose(1, 3)  # Transpose right before and after Conv
            x = self.output_conv(x)
            x = x.transpose(1, 3)
            x = torch.reshape(x, (bsz, -1, self.vocab_size))
            return x

        self.output_conv = nn.Conv2d(self.dim, self.vocab_size, 1, bias=False)
        self.output_conv.weight.data.copy_(self.output.weight[:, :, None, None])

        del self.output
        self.output = forward_output_conv

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind * self.n_kv_heads
                offset_v = self.n_layers * self.n_kv_heads + offset_k
                k_caches = args[offset_k : offset_k + self.n_kv_heads]
                v_caches = args[offset_v : offset_v + self.n_kv_heads]
            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self, use_kv_cache=True):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )

        atten_mask = torch.full((self.ar_len, self.ar_len), torch.tensor(-255.0))
        mask_cond = torch.arange(atten_mask.size(-1))
        atten_mask.masked_fill_(
            mask_cond < (mask_cond + 1).view(atten_mask.size(-1), 1), 0
        )
        if self.max_seq_len != self.ar_len:
            atten_mask = torch.cat(
                [
                    torch.ones(self.ar_len, self.max_seq_len - self.ar_len) * -255.0,
                    atten_mask,
                ],
                dim=-1,
            )
        atten_mask = atten_mask[None, :, :].expand(
            self.max_batch_size, self.ar_len, self.max_seq_len
        )
        if use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(self.n_layers):
                for _ in range(self.n_kv_heads):
                    # transpose first to decrease the runtime efforts
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.head_dim,
                            self.max_seq_len - self.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.max_seq_len - self.ar_len,
                            self.head_dim,
                        )
                    )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )

    def get_metadata(self):
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
        }


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    # The implementation of RoPE in HuggingFace processes query and key with two half instead of interleaved way.
    # The main difference is stride in StrideSlice op. For interleaved way, stride is two which is not friendly for HTP backend.
    # Ref: https://github.com/huggingface/transformers/issues/25199
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # broadcast for batch_prefill mode input x
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, None, :, :]
        freqs_sin = freqs_sin[None, None, :, :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for Qwen2.5-VL model.
    Based on official HuggingFace implementation.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Qwen2_5_VLRMSNorm(nn.Module):
    """RMS Normalization implementation for Qwen2.5-VL."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2_5_VLMLP(nn.Module):
    """
    Multi-Layer Perceptron for Qwen2.5-VL model.
    Based on official HuggingFace implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5_VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified for Qwen2.5-VL model with GQA support.
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLDecoderLayer(nn.Module):
    """
    Transformer decoder layer for Qwen2.5-VL model.
    Based on official HuggingFace implementation.
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2_5_VLAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2_5_VLMLP(config)
        self.input_layernorm = Qwen2_5_VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2_5_VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen2_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2_5_VLDecoderLayer`]
    """

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = Qwen2_5_VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Static kv cache
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(0)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel):
    """
    The Qwen2.5-VL Model for Conditional Generation.
    Based on official HuggingFace implementation (simplified for static export).
    """

    def __init__(
        self,
        config,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        ):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ar_len = ar_len
        self.bos_id = config.bos_token_id
        self.eos_id = config.eos_token_id
        self.dim = config.hidden_size
        self.output_cache = output_cache
        self.output_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.n_bos = 1
        self.n_eos = 1
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
        }

    def get_example_inputs(self, use_kv_cache=True):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )

        atten_mask = torch.full((self.ar_len, self.ar_len), torch.tensor(-255.0))
        mask_cond = torch.arange(atten_mask.size(-1))
        atten_mask.masked_fill_(
            mask_cond < (mask_cond + 1).view(atten_mask.size(-1), 1), 0
        )
        if self.max_seq_len != self.ar_len:
            atten_mask = torch.cat(
                [
                    torch.ones(self.ar_len, self.max_seq_len - self.ar_len) * -255.0,
                    atten_mask,
                ],
                dim=-1,
            )
        atten_mask = atten_mask[None, :, :].expand(
            self.max_batch_size, self.ar_len, self.max_seq_len
        )
        if use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(self.n_layers):
                for _ in range(self.n_kv_heads):
                    # transpose first to decrease the runtime efforts
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.head_dim,
                            self.max_seq_len - self.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.max_seq_len - self.ar_len,
                            self.head_dim,
                        )
                    )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )
        
    def get_metadata(self):
        # TODO: modify this when enabling LLAMA 7B
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": self.bos_id,
            "get_eos_id": self.eos_id,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
        }
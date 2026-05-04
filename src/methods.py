import types
from typing import Callable

import torch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def _kv_reduce_mean(x: torch.Tensor, kv_heads: int) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    if kv_heads <= 0 or kv_heads > heads:
        raise ValueError(f"kv_heads must be in [1, {heads}], got {kv_heads}")
    if heads % kv_heads != 0:
        raise ValueError(f"num_heads={heads} must be divisible by kv_heads={kv_heads}")
    group_size = heads // kv_heads
    x = x.view(batch, kv_heads, group_size, seq_len, dim).mean(dim=2)
    return x


def _kv_expand_repeat(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, kv_heads, seq_len, dim = x.shape
    if num_heads % kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by kv_heads={kv_heads}")
    group_size = num_heads // kv_heads
    return x.repeat_interleave(group_size, dim=1)


def _build_gqa_forward() -> Callable:
    def gqa_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        head_mask: torch.FloatTensor | None = None,
        layer_past=None,
        output_attentions: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: FlashAttentionKwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        num_heads = query_states.shape[1]
        kv_heads = self._gqa_kv_heads
        key_states = _kv_reduce_mean(key_states, kv_heads)
        value_states = _kv_reduce_mean(value_states, kv_heads)

        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = _kv_expand_repeat(key_states, num_heads)
        value_states = _kv_expand_repeat(value_states, num_heads)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            head_mask=head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights

    return gqa_forward


def enable_gqa(model, kv_heads: int) -> None:
    gqa_forward = _build_gqa_forward()
    for layer in model.gpt_neox.layers:
        attn = layer.attention
        attn._gqa_kv_heads = kv_heads
        attn.forward = types.MethodType(gqa_forward, attn)


def choose_attn_impl(mode: str, device: str) -> str:
    if "flash" not in mode:
        return "sdpa"
    if device != "cuda":
        return "sdpa"
    return "flash_attention_2"


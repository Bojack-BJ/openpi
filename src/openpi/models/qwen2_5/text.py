from collections.abc import Sequence
from typing import TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
from openpi.models.qwen2_5 import rotary as qwen_rotary
from openpi.models.vlm_backbone_config import Config
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


QWEN2_5_VL_VOCAB_SIZE = 151_936


@at.typecheck
class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x * jnp.reciprocal(jnp.sqrt(var + self.eps))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))
            return (normed * (1 + scale)).astype(dtype), None

        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        return (normed * (1 + scale) + shift).astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        return self.input_embedding_table[(x,)]


@at.typecheck
class Attention(nn.Module):
    configs: Sequence[Config]
    rope_theta: float = qwen_rotary.QWEN2_ROPE_THETA
    mrope_section: tuple[int, int, int] | None = None

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        assert self.configs[0].num_heads % self.configs[0].num_kv_heads == 0

        dtype = next(x.dtype for x in xs if x is not None)
        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            q = lora.Einsum(
                shape=(config.num_heads, config.width, config.head_dim),
                name=_name("q_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BTD,NDH->BTNH", x)
            k = lora.Einsum(
                shape=(config.num_kv_heads, config.width, config.head_dim),
                name=_name("k_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BSD,KDH->BSKH", x)
            v = lora.Einsum(
                shape=(config.num_kv_heads, config.width, config.head_dim),
                name=_name("v_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BSD,KDH->BSKH", x)
            qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
        q, k = qwen_rotary.apply_rotary_embedding(
            q,
            k,
            positions=positions,
            theta=self.rope_theta,
            mrope_section=self.mrope_section,
        )
        q *= self.configs[0].head_dim**-0.5

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
        cached_k = k
        cached_v = v

        num_kv_heads = self.configs[0].num_kv_heads
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
        k = qwen_rotary.repeat_kv(k, self.configs[0].num_heads // num_kv_heads)
        v = qwen_rotary.repeat_kv(v, self.configs[0].num_heads // num_kv_heads)
        k = einops.rearrange(k, "B S (K G) H -> B S K G H", K=num_kv_heads)
        v = einops.rearrange(v, "B S (K G) H -> B S K G H", K=num_kv_heads)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        logits = jnp.einsum("BTKGH,BSKGH->BKGTS", q, k, preferred_element_type=jnp.float32)
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKGH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("o_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (cached_k, cached_v)


@at.typecheck
class FeedForward(nn.Module):
    features: int
    hidden_dim: int
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        gate_proj = lora.Einsum(
            shape=(self.features, self.hidden_dim),
            name="gate_proj",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            lora_config=self.lora_config,
        )("BTD,DH->BTH", x)
        up_proj = lora.Einsum(
            shape=(self.features, self.hidden_dim),
            name="up_proj",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            lora_config=self.lora_config,
        )("BTD,DH->BTH", x)
        activations = nn.silu(gate_proj) * up_proj
        down_proj = lora.Einsum(
            shape=(self.hidden_dim, self.features),
            name="down_proj",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            lora_config=self.lora_config,
        )("BTH,HD->BTD", activations)
        return down_proj.astype(dtype)


@at.typecheck
class Block(nn.Module):
    configs: tuple[Config, ...]
    rope_theta: float = qwen_rotary.QWEN2_ROPE_THETA
    mrope_section: tuple[int, int, int] | None = None
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(
            configs=self.configs,
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
            name="attn",
        )

        pre_attn = []
        attn_gates = []
        for i, x in enumerate(xs):
            if x is None:
                pre_attn.append(None)
                attn_gates.append(None)
                continue
            normed, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
            pre_attn.append(normed)
            attn_gates.append(gate)
        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, attn_gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        ffw_gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                ffw_gates.append(None)
                continue
            normed, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
            ff_out = FeedForward(
                features=config.width,
                hidden_dim=config.mlp_dim,
                name=_name("mlp", i),
                lora_config=config.lora_configs.get("ffn"),
            )(normed)
            out.append(ff_out)
            ffw_gates.append(gate)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, ffw_gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)
        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    configs: Sequence[Config]
    embed_dtype: str
    vocab_size: int = QWEN2_5_VL_VOCAB_SIZE
    rope_theta: float = qwen_rotary.QWEN2_ROPE_THETA
    mrope_section: tuple[int, int, int] | None = None
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=self.vocab_size,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=tuple(self.configs),
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Array,
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        positions = qwen_rotary.repeat_text_positions(positions)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        total_tokens = len(self.configs)
        self(
            [jnp.zeros((1, 1, c.width), dtype=jnp.dtype(self.embed_dtype)) for c in self.configs],
            jnp.broadcast_to(jnp.arange(total_tokens, dtype=jnp.int32)[None, :], (1, total_tokens)),
            jnp.ones((1, total_tokens, total_tokens), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width), dtype=jnp.dtype(self.embed_dtype)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _name(name, i):
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate

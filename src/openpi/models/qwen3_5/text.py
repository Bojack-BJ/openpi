from collections.abc import Sequence
from typing import Any
from typing import TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
from openpi.models.qwen3_5 import rotary as qwen_rotary
from openpi.models.vlm_backbone_config import Config
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


QWEN3_5_VOCAB_SIZE = 248_320
QWEN3_5_ROPE_THETA = 10_000_000.0


@at.typecheck
class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x * jnp.reciprocal(jnp.sqrt(var + self.eps))
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))
        return (normed * (1 + scale)).astype(dtype)


@at.typecheck
class RMSNormGated(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, gate):
        gated = x * nn.silu(gate.astype(x.dtype))
        dtype = gated.dtype
        var = jnp.mean(jnp.square(gated.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = gated * jnp.reciprocal(jnp.sqrt(var + self.eps))
        scale = self.param("scale", nn.initializers.zeros_init(), (gated.shape[-1],))
        return (normed * (1 + scale)).astype(dtype)


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


def _masked_token_mask(attn_mask: at.Bool[at.Array, "b t s"]) -> at.Bool[at.Array, "b t"]:
    return jnp.any(attn_mask, axis=-1)


def _zero_invalid_tokens(hidden_states: at.Array, token_mask: at.Array) -> at.Array:
    return hidden_states * token_mask[..., None].astype(hidden_states.dtype)


class DepthwiseShortConv1D(nn.Module):
    channels: int
    kernel_size: int

    @nn.compact
    def __call__(self, x, *, cache=None):
        kernel = self.param("kernel", nn.initializers.normal(stddev=0.02), (self.kernel_size, self.channels))

        batch_size = x.shape[0]
        if cache is None:
            cache = jnp.zeros((batch_size, max(self.kernel_size - 1, 0), self.channels), dtype=x.dtype)

        def step(state, x_t):
            window = jnp.concatenate([state, x_t[:, None, :]], axis=1)
            y_t = jnp.sum(window * kernel[::-1][None, :, :].astype(x.dtype), axis=1)
            y_t = nn.silu(y_t)
            next_state = window[:, 1:, :] if self.kernel_size > 1 else state
            return next_state, y_t

        next_cache, ys = jax.lax.scan(step, cache, jnp.swapaxes(x, 0, 1))
        return jnp.swapaxes(ys, 0, 1), next_cache


@at.typecheck
class GatedAttention(nn.Module):
    configs: Sequence[Config]
    rope_theta: float = qwen_rotary.QWEN3_5_ROPE_THETA
    partial_rotary_factor: float = qwen_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR
    mrope_section: tuple[int, int, int] | None = None

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        assert self.configs[0].num_heads % self.configs[0].num_kv_heads == 0

        dtype = next(x.dtype for x in xs if x is not None)
        token_mask = _masked_token_mask(attn_mask)
        qkgs = []
        cursor = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            x = _zero_invalid_tokens(x, token_mask[:, cursor : cursor + x.shape[1]])
            cursor += x.shape[1]
            qg = lora.Einsum(
                shape=(config.num_heads, config.width, 2 * config.head_dim),
                name=_name("qg_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BTD,NDH->BTNH", x)
            q, gate = jnp.split(qg, 2, axis=-1)
            q = RMSNorm(name=_name("q_norm", i))(q)

            k = lora.Einsum(
                shape=(config.num_kv_heads, config.width, config.head_dim),
                name=_name("k_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BTD,KDH->BTKH", x)
            k = RMSNorm(name=_name("k_norm", i))(k)
            v = lora.Einsum(
                shape=(config.num_kv_heads, config.width, config.head_dim),
                name=_name("v_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                lora_config=config.lora_configs.get("attn"),
            )("BTD,KDH->BTKH", x)
            qkgs.append((q, k, v, gate))

        q, k, v, gate = (jnp.concatenate(y, axis=1) for y in zip(*qkgs, strict=True))
        q, k = qwen_rotary.apply_rotary_embedding(
            q,
            k,
            positions=positions,
            theta=self.rope_theta,
            mrope_section=self.mrope_section,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        q *= self.configs[0].head_dim**-0.5

        cache_k = cache_v = None
        if kv_cache is not None:
            cache_k, cache_v, _, _ = kv_cache
        if cache_k is not None and cache_v is not None:
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
        cached_k = k
        cached_v = v

        k = qwen_rotary.repeat_kv(k, self.configs[0].num_heads // self.configs[0].num_kv_heads)
        v = qwen_rotary.repeat_kv(v, self.configs[0].num_heads // self.configs[0].num_kv_heads)

        mask = attn_mask[:, None, :, :]
        logits = jnp.einsum("BTNH,BSNH->BNTS", q, k, preferred_element_type=jnp.float32)
        masked_logits = jnp.where(mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        probs = probs * jnp.any(mask, axis=-1, keepdims=True).astype(dtype)

        encoded = jnp.einsum("BNTS,BSNH->BTNH", probs, v)
        encoded = encoded * jax.nn.sigmoid(gate).astype(encoded.dtype)
        encoded = encoded.reshape(encoded.shape[0], encoded.shape[1], -1)

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]
            out_einsum = lora.Einsum(
                shape=(config.num_heads, config.head_dim, config.width),
                name=_name("o_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                lora_config=config.lora_configs.get("attn"),
            )
            out.append(
                out_einsum(
                    "BTNH,NHD->BTD",
                    encoded[:, start:end].reshape(x.shape[0], x.shape[1], config.num_heads, config.head_dim),
                )
            )
            start = end

        return out, (cached_k, cached_v, None, None)


@at.typecheck
class GatedDeltaNet(nn.Module):
    configs: Sequence[Config]
    rope_theta: float = qwen_rotary.QWEN3_5_ROPE_THETA
    partial_rotary_factor: float = qwen_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR
    mrope_section: tuple[int, int, int] | None = None

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        config = self.configs[0]
        assert all(c.linear_num_key_heads == config.linear_num_key_heads for c in self.configs)
        assert all(c.linear_num_value_heads == config.linear_num_value_heads for c in self.configs)
        assert all(c.linear_key_head_dim == config.linear_key_head_dim for c in self.configs)
        assert all(c.linear_value_head_dim == config.linear_value_head_dim for c in self.configs)

        if config.linear_num_key_heads is None or config.linear_num_value_heads is None:
            raise ValueError("Qwen3.5 linear-attention config is missing linear head counts.")
        if config.linear_key_head_dim is None or config.linear_value_head_dim is None:
            raise ValueError("Qwen3.5 linear-attention config is missing linear head dims.")
        if config.linear_conv_kernel_dim is None:
            raise ValueError("Qwen3.5 linear-attention config is missing convolution kernel size.")

        num_k_heads = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        key_dim = config.linear_key_head_dim
        value_dim = config.linear_value_head_dim
        conv_hidden = num_k_heads * key_dim * 2 + num_v_heads * value_dim
        dtype = next(x.dtype for x in xs if x is not None)
        token_mask = _masked_token_mask(attn_mask)

        projected = []
        z_chunks = []
        b_chunks = []
        a_chunks = []
        cursor = 0
        for i, (x, branch_config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            x = _zero_invalid_tokens(x, token_mask[:, cursor : cursor + x.shape[1]])
            cursor += x.shape[1]
            qkv = lora.Einsum(
                shape=(branch_config.width, conv_hidden),
                name=_name("in_proj_qkv", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
                lora_config=branch_config.lora_configs.get("attn"),
            )("BTD,DH->BTH", x)
            z = lora.Einsum(
                shape=(branch_config.width, num_v_heads * value_dim),
                name=_name("in_proj_z", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
                lora_config=branch_config.lora_configs.get("attn"),
            )("BTD,DH->BTH", x)
            beta = lora.Einsum(
                shape=(branch_config.width, num_v_heads),
                name=_name("in_proj_b", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
                lora_config=branch_config.lora_configs.get("attn"),
            )("BTD,DH->BTH", x)
            alpha = lora.Einsum(
                shape=(branch_config.width, num_v_heads),
                name=_name("in_proj_a", i),
                init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
                lora_config=branch_config.lora_configs.get("attn"),
            )("BTD,DH->BTH", x)
            projected.append(qkv)
            z_chunks.append(z)
            b_chunks.append(beta)
            a_chunks.append(alpha)

        qkv = jnp.concatenate(projected, axis=1)
        z = jnp.concatenate(z_chunks, axis=1).reshape(qkv.shape[0], qkv.shape[1], num_v_heads, value_dim)
        beta = jax.nn.sigmoid(jnp.concatenate(b_chunks, axis=1))
        a = jnp.concatenate(a_chunks, axis=1)

        _, _, conv_cache, recurrent_cache = kv_cache if kv_cache is not None else (None, None, None, None)
        qkv, conv_cache = DepthwiseShortConv1D(
            channels=conv_hidden,
            kernel_size=config.linear_conv_kernel_dim,
            name="short_conv",
        )(qkv, cache=conv_cache)

        split_sizes = (
            num_k_heads * key_dim,
            num_k_heads * key_dim,
            num_v_heads * value_dim,
        )
        split_points = (split_sizes[0], split_sizes[0] + split_sizes[1])
        query, key, value = jnp.split(qkv, split_points, axis=-1)
        query = query.reshape(qkv.shape[0], qkv.shape[1], num_k_heads, key_dim)
        key = key.reshape(qkv.shape[0], qkv.shape[1], num_k_heads, key_dim)
        value = value.reshape(qkv.shape[0], qkv.shape[1], num_v_heads, value_dim)

        query, key = qwen_rotary.apply_rotary_embedding(
            query,
            key,
            positions=positions,
            theta=self.rope_theta,
            mrope_section=self.mrope_section,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        if num_v_heads % num_k_heads != 0:
            raise ValueError(f"Qwen3.5 linear attention expects value heads to be divisible by key heads: {num_v_heads}/{num_k_heads}")
        repeat_factor = num_v_heads // num_k_heads
        query = qwen_rotary.repeat_kv(query, repeat_factor)
        key = qwen_rotary.repeat_kv(key, repeat_factor)

        decay_scale = jnp.exp(self.param("A_log", nn.initializers.zeros_init(), (num_v_heads,))).astype(dtype)
        dt_bias = self.param("dt_bias", nn.initializers.zeros_init(), (num_v_heads,)).astype(dtype)
        g = -decay_scale[None, None, :] * nn.softplus(a.astype(dtype) + dt_bias[None, None, :])

        valid = token_mask[:, : qkv.shape[1]].astype(dtype)
        query = query * valid[:, :, None, None]
        key = key * valid[:, :, None, None]
        value = value * valid[:, :, None, None]
        z = z * valid[:, :, None, None]
        beta = beta.astype(dtype) * valid[:, :, None]
        g = jnp.where(valid[:, :, None].astype(bool), g, 0.0)

        outputs, recurrent_cache = _gated_delta_recurrence(
            query,
            key,
            value,
            g,
            beta,
            initial_state=recurrent_cache,
        )
        outputs = RMSNormGated(name="norm")(outputs, z).reshape(outputs.shape[0], outputs.shape[1], -1)

        out = []
        start = 0
        for i, (x, branch_config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]
            out_proj = lora.Einsum(
                shape=(num_v_heads, value_dim, branch_config.width),
                name=_name("out_proj", i),
                init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                lora_config=branch_config.lora_configs.get("attn"),
            )
            branch = outputs[:, start:end].reshape(outputs.shape[0], x.shape[1], num_v_heads, value_dim)
            out.append(out_proj("BTNH,NHD->BTD", branch))
            start = end

        return out, (None, None, conv_cache, recurrent_cache)


def _gated_delta_recurrence(query, key, value, g, beta, *, initial_state):
    query = _l2_normalize(query) * (query.shape[-1] ** -0.5)
    key = _l2_normalize(key)
    batch_size, _, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros((batch_size, num_heads, key_dim, value_dim), dtype=value.dtype)
    state_dtype = initial_state.dtype
    compute_dtype = jnp.float32 if state_dtype in (jnp.bfloat16, jnp.float16) else state_dtype

    def step(state, inputs):
        q_t, k_t, v_t, g_t, beta_t = inputs
        state_f = state.astype(compute_dtype)
        q_t = q_t.astype(compute_dtype)
        k_t = k_t.astype(compute_dtype)
        v_t = v_t.astype(compute_dtype)
        beta_t = beta_t.astype(compute_dtype)
        g_t = g_t.astype(compute_dtype)

        state_f = state_f * jnp.exp(g_t)[..., None, None]
        kv_mem = jnp.einsum("bhkv,bhk->bhv", state_f, k_t)
        state_f = state_f + jnp.einsum("bhk,bhv->bhkv", k_t, beta_t[..., None] * (v_t - kv_mem))
        out_t = jnp.einsum("bhkv,bhk->bhv", state_f, q_t)
        return state_f.astype(state_dtype), out_t.astype(value.dtype)

    final_state, outputs = jax.lax.scan(
        step,
        initial_state,
        (
            jnp.swapaxes(query, 0, 1),
            jnp.swapaxes(key, 0, 1),
            jnp.swapaxes(value, 0, 1),
            jnp.swapaxes(g, 0, 1),
            jnp.swapaxes(beta, 0, 1),
        ),
    )
    return jnp.swapaxes(outputs, 0, 1), final_state


def _l2_normalize(x, eps: float = 1e-6):
    return x / jnp.maximum(jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True), eps)


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


LayerCache: TypeAlias = tuple[Any, Any, Any, Any]


@at.typecheck
class DecoderLayer(nn.Module):
    configs: tuple[Config, ...]
    layer_type: str
    rope_theta: float = qwen_rotary.QWEN3_5_ROPE_THETA
    partial_rotary_factor: float = qwen_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR
    mrope_section: tuple[int, int, int] | None = None

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        del deterministic
        if any(cond is not None for cond in adarms_cond):
            raise NotImplementedError("JAX Qwen3.5 text blocks do not support pi05/AdaRMS conditioning yet.")

        xs = sharding.activation_sharding_constraint(xs)
        pre_attn = [RMSNorm(name=_name("pre_attention_norm", i))(x) if x is not None else None for i, x in enumerate(xs)]
        pre_attn = sharding.activation_sharding_constraint(pre_attn)

        if self.layer_type == "full_attention":
            post_attn, kv_cache = GatedAttention(
                configs=self.configs,
                rope_theta=self.rope_theta,
                partial_rotary_factor=self.partial_rotary_factor,
                mrope_section=self.mrope_section,
                name="self_attn",
            )(pre_attn, positions, attn_mask, kv_cache)
        elif self.layer_type == "linear_attention":
            post_attn, kv_cache = GatedDeltaNet(
                configs=self.configs,
                rope_theta=self.rope_theta,
                partial_rotary_factor=self.partial_rotary_factor,
                mrope_section=self.mrope_section,
                name="self_attn",
            )(pre_attn, positions, attn_mask, kv_cache)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {self.layer_type}")

        xs = [x + y if x is not None else None for x, y in zip(xs, post_attn, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                continue
            normed = RMSNorm(name=_name("pre_ffw_norm", i))(x)
            ff_out = FeedForward(
                features=config.width,
                hidden_dim=config.mlp_dim,
                name=_name("mlp", i),
                lora_config=config.lora_configs.get("ffn"),
            )(normed)
            out.append(ff_out)

        out = sharding.activation_sharding_constraint(out)
        xs = [x + y if x is not None else None for x, y in zip(xs, out, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)
        return xs, kv_cache


GroupCache: TypeAlias = tuple[LayerCache, LayerCache, LayerCache, LayerCache]


@at.typecheck
class DecoderLayerGroup(nn.Module):
    configs: tuple[Config, ...]
    layer_types: tuple[str, str, str, str]
    rope_theta: float = qwen_rotary.QWEN3_5_ROPE_THETA
    partial_rotary_factor: float = qwen_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR
    mrope_section: tuple[int, int, int] | None = None

    def setup(self):
        self.layers = tuple(
            DecoderLayer(
                configs=self.configs,
                layer_type=layer_type,
                rope_theta=self.rope_theta,
                partial_rotary_factor=self.partial_rotary_factor,
                mrope_section=self.mrope_section,
                name=f"layers_{i}",
            )
            for i, layer_type in enumerate(self.layer_types)
        )

    @at.typecheck
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        if kv_cache is None:
            kv_cache = tuple(None for _ in self.layers)

        next_cache = []
        for layer, layer_cache in zip(self.layers, kv_cache, strict=True):
            xs, layer_cache = layer(xs, layer_cache, positions, attn_mask, adarms_cond, deterministic)
            next_cache.append(layer_cache)
        return xs, tuple(next_cache)


KVCache: TypeAlias = Any


@at.typecheck
class Module(nn.Module):
    configs: Sequence[Config]
    embed_dtype: str
    vocab_size: int = QWEN3_5_VOCAB_SIZE
    rope_theta: float = qwen_rotary.QWEN3_5_ROPE_THETA
    partial_rotary_factor: float = qwen_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR
    mrope_section: tuple[int, int, int] | None = None

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        layer_types = self.configs[0].layer_types
        if layer_types is None:
            raise ValueError("Qwen3.5 text module requires explicit `layer_types` in the backbone config.")
        if len(layer_types) != self.configs[0].depth:
            raise ValueError(f"Qwen3.5 layer_types length {len(layer_types)} does not match depth {self.configs[0].depth}")
        if self.configs[0].depth % 4 != 0:
            raise ValueError(f"Qwen3.5 depth {self.configs[0].depth} must be divisible by the 4-layer hybrid pattern.")

        layer_group_types = tuple(layer_types[:4])
        for start in range(0, self.configs[0].depth, 4):
            if tuple(layer_types[start : start + 4]) != layer_group_types:
                raise ValueError("Qwen3.5 text module currently requires a repeated 4-layer hybrid pattern.")

        self.embedder = Embedder(
            vocab_size=self.vocab_size,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            DecoderLayerGroup,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth // 4,
        )(
            configs=tuple(self.configs),
            layer_types=layer_group_types,
            rope_theta=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            mrope_section=self.mrope_section,
        )
        self.final_norms = tuple(RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs)))

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
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        xs, kv_cache = self.layers(list(embedded), kv_cache, positions, mask, adarms_cond, deterministic)
        return [f(e) if e is not None else e for f, e in zip(self.final_norms, xs, strict=True)], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        if any(use_adarms):
            raise NotImplementedError("JAX Qwen3.5 text blocks do not support pi05/AdaRMS conditioning yet.")
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        total_tokens = len(self.configs)
        self(
            [jnp.zeros((1, 1, c.width), dtype=jnp.dtype(self.embed_dtype)) for c in self.configs],
            jnp.broadcast_to(jnp.arange(total_tokens, dtype=jnp.int32)[None, :], (1, total_tokens)),
            jnp.ones((1, total_tokens, total_tokens), dtype=bool),
            adarms_cond=[None for _ in self.configs],
        )


def _name(name, i):
    if i == 0:
        return name
    return f"{name}_{i}"


__all__ = [
    "DecoderLayer",
    "Embedder",
    "FeedForward",
    "GatedAttention",
    "GatedDeltaNet",
    "KVCache",
    "Module",
    "QWEN3_5_ROPE_THETA",
    "QWEN3_5_VOCAB_SIZE",
    "RMSNorm",
    "RMSNormGated",
]

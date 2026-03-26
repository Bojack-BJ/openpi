# OpenPI 里的 Joint Attention 是怎么工作的

这份文档解释当前 OpenPI 里 `pi0` 模型的 `joint attention` 机制，分别说明：

- prefix / suffix token 是怎么定义的
- JAX 路径里 joint attention 的实现位置和执行顺序
- PyTorch 路径里 joint attention 的实现位置和执行顺序
- 推理时为什么要先跑 prefix 再复用 KV cache
- 当前 Qwen 接入是怎么复用这套契约的

本文讨论的是当前仓库里的 `pi0` 主通路，不展开 `pi0_fast`。

## 1. 先说结论：这不是 cross-attention，而是“共享 self-attention + 分专家 FFN”

这里的 `joint attention` 不是“prefix 做 encoder，suffix 做 decoder，然后 suffix cross-attend prefix”。

它做的是：

1. prefix expert 和 suffix expert 各自先做自己的 `Q/K/V` 投影。
2. 把两边的 token 在 sequence 维上拼起来，形成一条“联合序列”。
3. 在这条联合序列上做一次共享 attention。
4. attention 输出再按原来的 token 区间切回 prefix / suffix。
5. 切回去以后，每一边再走自己的 `o_proj`、residual、norm、MLP。

所以它的本质是：

- attention 是 joint 的
- block 之后的参数是 expert-specific 的

JAX 版最核心的代码在：

- `src/openpi/models/gemma.py:96-181`
- `src/openpi/models/gemma.py:225-265`

PyTorch 版最核心的代码在：

- `src/openpi/models_pytorch/gemma_pytorch.py:173-254`

Qwen 版沿用同样的结构，代码在：

- `src/openpi/models_pytorch/qwen2_vl_pytorch.py:227-295`

## 2. prefix / suffix 到底是什么

在 OpenPI 里，`prefix` / `suffix` 不是字符串层面的“前缀”和“后缀”，而是模型内部的两段 token block。

### 2.1 prefix

prefix 由两部分组成：

- image tokens
- text prompt tokens

JAX 里对应：

- `src/openpi/models/pi0.py:107-138`

PyTorch 里对应：

- `src/openpi/models_pytorch/pi0_pytorch.py:211-260`

JAX 的关键代码：

```python
for name in obs.images:
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
    tokens.append(image_tokens)
    ar_mask += [False] * image_tokens.shape[1]

if obs.tokenized_prompt is not None:
    tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
    tokens.append(tokenized_inputs)
    ar_mask += [False] * tokenized_inputs.shape[1]
```

这里 `ar_mask=False` 的意思是：整段 prefix 处在同一个 attention block 里，内部是全连接注意力。

### 2.2 suffix

`pi0` 下的 suffix 由两部分组成：

- 1 个 state token
- `action_horizon` 个 action tokens

JAX 里对应：

- `src/openpi/models/pi0.py:141-187`

PyTorch 里对应：

- `src/openpi/models_pytorch/pi0_pytorch.py:262-339`

JAX 的关键代码：

```python
state_token = self.state_proj(obs.state)[:, None, :]
ar_mask += [True]

action_tokens = self.action_in_proj(noisy_actions)
...
ar_mask += [True] + ([False] * (self.action_horizon - 1))
```

这里要注意一个容易误解的点：

- state token 自己是一个 block
- 全部 action tokens 是另一个 block
- action block 内部是全 attention，不是严格逐 token causal

也就是说，这里的 action horizon token 更像“一组并行 denoising token”，不是语言模型那种逐 token 自回归输出。

## 3. mask 是怎么把 prefix / suffix 变成 joint attention 规则的

这一步是整套机制的入口。

JAX:

- `src/openpi/models/pi0.py:20-45`

PyTorch:

- `src/openpi/models_pytorch/pi0_pytorch.py:52-81`

JAX 版本：

```python
mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
cumsum = jnp.cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
return jnp.logical_and(attn_mask, valid_mask)
```

它的语义不是普通的“下三角 causal mask”，而是“按 block 编号做可见性”：

- `mask_ar=False/0`：和前一个 token 处在同一个 block
- `mask_ar=True/1`：从这个 token 开始，进入一个新的 block
- query 只能看到 `block_id <= 自己` 的 key

因此在 `pi0` 里会得到这条规则：

- prefix 可以互相看见
- state 可以看见所有 prefix 和自己
- actions 可以看见所有 prefix、state 和全部 action
- prefix 不能回头看 suffix

这就是 joint attention 生效的“调度规则”。

## 4. JAX 路径：joint attention 是怎么一层一层执行的

## 4.1 先把两个 expert 初始化出来

入口在：

- `src/openpi/models/pi0.py:67-104`

这里会构造一个 `gemma.Module(configs=[vlm_backbone_config, action_expert_config])`：

```python
llm = nnx_bridge.ToNNX(
    _gemma.Module(
        configs=[vlm_backbone_config, action_expert_config],
        embed_dtype=config.dtype,
        adarms=config.pi05,
    )
)
```

`configs` 里有两个 expert：

- expert 0: prefix 对应的 VLM/text backbone
- expert 1: suffix 对应的 action expert

真正的 joint attention 核心是在 `gemma.Module -> Block -> Attention` 这一层层调用里。

## 4.2 训练时：prefix 和 suffix 一次性拼进同一个 forward

训练主入口在：

- `src/openpi/models/pi0.py:189-215`

关键代码：

```python
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
attn_mask = make_attn_mask(input_mask, ar_mask)
positions = jnp.cumsum(input_mask, axis=1) - 1
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
)
```

重点有两个：

1. `inputs` 不是单个序列，而是 `[prefix_tokens, suffix_tokens]`
2. attention mask 是对“联合序列”定义的

## 4.3 `gemma.Module` 把两个 expert 一起跑

入口在：

- `src/openpi/models/gemma.py:321-343`

关键逻辑：

```python
embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)
return [
    f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
], kv_cache
```

也就是说：

- `embedded` 是一个长度为 2 的列表
- 每层同时处理两个 expert 的 token
- 最终再分别过自己的 final norm

## 4.4 `Block`：先做联合 attention，再各自做自己的 FFN

代码在：

- `src/openpi/models/gemma.py:225-265`

结构非常清楚：

```python
for i, x in enumerate(xs):
    if x is not None:
        x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
...
post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
...
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
        x = lora.FeedForward(...)(x)
...
xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
```

可以把一层理解成：

1. prefix / suffix 各自做自己的 pre-attention norm
2. 进入同一个 `Attention`
3. 从 attention 出来以后，各自 residual 回自己的 hidden states
4. 然后各自跑自己的 FFN

所以“共享”的只有 attention 部分，不共享 FFN。

## 4.5 `Attention`：真正的 joint attention 发生在这里

代码在：

- `src/openpi/models/gemma.py:96-181`

先看最关键的三步。

### 第一步：每个 expert 各自算自己的 Q / K / V

```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    ...
    q = q_einsum("BTD,NDH->BTNH", x)
    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
    qkvs.append((q, k, v))
```

### 第二步：沿着 sequence 维把两边拼起来

```python
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

这一步就是“joint”的定义本身。

- prefix token 的 q/k/v
- suffix token 的 q/k/v

现在变成了一条联合序列上的 q/k/v。

### 第三步：在联合序列上做 attention，然后再切回去

```python
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
...
end = start + x.shape[1]
out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
```

这里发生了三件事：

1. attention 的 key/value 已经是 prefix + suffix 的并集
2. mask 决定哪些 token 可以互相看见
3. 结果按原始长度区间切回 prefix / suffix

这也是为什么前面要求两个 expert 的 attention 几何必须兼容：

- `head_dim`
- `num_heads`
- `num_kv_heads`

JAX 里这点在一开始就有断言：

- `src/openpi/models/gemma.py:97-100`

## 5. JAX 推理：为什么先跑 prefix，再只跑 suffix

代码在：

- `src/openpi/models/pi0.py:217-280`

推理阶段的思路是：

1. 先把 prefix 全部过一遍，填满 KV cache
2. 后续每个 denoise step 只重新算 suffix
3. suffix 通过 cache 去看 prefix，不需要重复编码 prefix

关键代码：

```python
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

然后每个 denoise step：

```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],
    mask=full_attn_mask,
    positions=positions,
    kv_cache=kv_cache,
    adarms_cond=[None, adarms_cond],
)
```

这里的 `full_attn_mask` 是“suffix 作为 query，可以看到 prefix cache + 当前 suffix”的可见性矩阵。

这套做法的收益很直接：

- 图像编码只做一次
- prefix transformer 只做一次
- 每个 denoise step 的成本只和 suffix 长度相关

## 6. PyTorch 路径：整体结构和 JAX 一样，但实现更手工

## 6.1 prefix / suffix 还是同样的拼法

prefix:

- `src/openpi/models_pytorch/pi0_pytorch.py:211-260`

suffix:

- `src/openpi/models_pytorch/pi0_pytorch.py:262-339`

mask:

- `src/openpi/models_pytorch/pi0_pytorch.py:52-81`

训练入口：

- `src/openpi/models_pytorch/pi0_pytorch.py:341-397`

核心代码：

```python
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(...)
pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
position_ids = torch.cumsum(pad_masks, dim=1) - 1
(_, suffix_out), _ = self.vlm_with_expert.forward(
    attention_mask=att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, suffix_embs],
    use_cache=False,
    adarms_cond=[None, adarms_cond],
)
```

思路和 JAX 完全一致：

- 先构造 prefix / suffix embedding
- 再构造联合 mask
- 然后把 `[prefix_embs, suffix_embs]` 一起送进 joint forward

## 6.2 `PaliGemmaWithExpertModel.forward()` 有三种模式

代码在：

- `src/openpi/models_pytorch/gemma_pytorch.py:107-297`

它实际上支持三种运行方式：

1. `inputs_embeds=[prefix, None]`
   只跑 prefix，用于推理前的 prefix cache 预填充
2. `inputs_embeds=[None, suffix]`
   只跑 suffix，用于推理阶段的 denoise step
3. `inputs_embeds=[prefix, suffix]`
   走 joint attention，用于训练

前两种直接调用 HF 原生模型；第三种才进入手写的 joint attention 路径。

## 6.3 PyTorch 的 joint attention 核心在 `compute_layer_complete`

代码在：

- `src/openpi/models_pytorch/gemma_pytorch.py:173-254`

这一段就是 Torch 版对 JAX `Attention + Block` 的手工翻译。

### 第一步：两边各自做 input norm，再各自投影 Q / K / V

```python
for i, hidden_states in enumerate(inputs_embeds):
    layer = models[i].layers[layer_idx]
    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
    ...
    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

### 第二步：拼成联合序列

```python
query_states = torch.cat(query_states, dim=2)
key_states = torch.cat(key_states, dim=2)
value_states = torch.cat(value_states, dim=2)
```

这里 `dim=2` 对应 sequence 维，因为当前张量布局是：

- `[batch, heads, seq, head_dim]`

### 第三步：对联合序列做 RoPE 和 attention

```python
cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
    query_states, key_states, cos, sin, unsqueeze_dim=1
)

att_output, _ = modeling_gemma.eager_attention_forward(
    self.paligemma.language_model.layers[layer_idx].self_attn,
    query_states,
    key_states,
    value_states,
    attention_mask,
    scaling,
)
```

### 第四步：切回 prefix / suffix，然后各自走自己的后处理

```python
end_pos = start_pos + hidden_states.shape[1]
out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
after_first_residual = out_emb.clone()
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
out_emb = layer.mlp(out_emb)
out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
outputs_embeds.append(out_emb)
```

这和 JAX 的语义是完全对齐的：

- joint attention 之后先切回各自的 token 段
- 再做各自的 `o_proj`
- 再做各自的 residual / norm / MLP

## 6.4 PyTorch 推理：也是先 prefix cache，再 suffix denoise

代码在：

- `src/openpi/models_pytorch/pi0_pytorch.py:399-485`

prefix cache 预填充：

```python
_, past_key_values = self.vlm_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
```

每一步 denoise：

```python
outputs_embeds, _ = self.vlm_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=past_key_values,
    inputs_embeds=[None, suffix_embs],
    use_cache=False,
    adarms_cond=[None, adarms_cond],
)
```

和 JAX 一样，这里的计算图是：

- prefix 不再重复算
- suffix 通过 cache 访问 prefix

## 7. Qwen 是怎么接进这套 joint attention 契约的

当前 Qwen 的实现不是“改成官方 Qwen2.5-VL 的完整多模态输入流”，而是：

- 保留 OpenPI 现有的 prefix / suffix 契约
- 把 prefix VLM backbone 换成 `Qwen2.5-VL`
- 把 suffix expert 换成 `Qwen2`
- joint attention 逻辑继续沿用当前 OpenPI 的实现方式

代码在：

- `src/openpi/models_pytorch/qwen2_vl_pytorch.py:185-319`

它的 joint attention 结构和 Gemma 版是同构的：

```python
query_states = torch.cat(query_states, dim=2)
key_states = torch.cat(key_states, dim=2)
value_states = torch.cat(value_states, dim=2)
...
attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(
    query_states.shape[-1]
)
...
out_emb = layer.self_attn.o_proj(attn_output[:, start_pos:end_pos])
hidden_states = residual + out_emb
...
hidden_states = layer.mlp(hidden_states)
hidden_states = residual + hidden_states
```

所以从架构角度看：

- Qwen 不是换了一种 attention 范式
- 只是把“参与这次 joint attention 的两套专家权重”换掉了

## 8. 你可以先按这个阅读顺序看代码

如果你想最快理解整条链路，建议按下面顺序读：

1. `src/openpi/models/pi0.py:189-215`
2. `src/openpi/models/pi0.py:107-187`
3. `src/openpi/models/gemma.py:225-265`
4. `src/openpi/models/gemma.py:96-181`
5. `src/openpi/models_pytorch/pi0_pytorch.py:341-397`
6. `src/openpi/models_pytorch/gemma_pytorch.py:173-254`

如果你想理解推理为什么能省算力，再看：

1. `src/openpi/models/pi0.py:234-268`
2. `src/openpi/models_pytorch/pi0_pytorch.py:409-479`

## 9. 一句话总结

OpenPI 的 `joint attention` 可以概括成一句话：

> prefix expert 和 suffix expert 各自产生 Q/K/V，在联合序列上共享一次 attention，再把结果切回各自 token 段，继续走各自的残差块和 MLP。

这也是为什么当前实现会强依赖两边的 attention 几何兼容，而不是随便把任意两个 backbone 直接拼起来。

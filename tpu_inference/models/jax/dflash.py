# Copyright 2025 Google LLC
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
"""DFlash draft model for speculative decoding on JAX/TPU."""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from transformers import Qwen3Config
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.kernels.flash_attention.kernel import (BlockSizes,
                                                          SegmentIds,
                                                          flash_attention)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (BaseWeightLoader,
                                                         get_default_maps,
                                                         load_hf_weights)
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

# vmem budget for the flash_attention Pallas kernel (128 MiB).
_FA_VMEM_LIMIT = 128 * 1024 * 1024


class DFlashAttention(nnx.Module):
    """DFlash cross+self attention with on-device KV cache.

    Each call:
      1. Projects Q from noise embeddings, K/V from [context, noise].
      2. Applies RoPE to Q and K.
      3. Expands K/V for GQA.
      4. Writes NEW K/V into the pre-allocated cache via dynamic_update_slice.
      5. Runs non-causal flash_attention over the full cache up to the valid
         length, using segment_ids to mask padding.
    """

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = get_mesh_shape_product(mesh,
                                               ShardingAxisName.MLP_TENSOR)
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.ATTN_HEAD, None, None)),
            rngs=rng,
        )

        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )

    def __call__(
        self,
        x_noise: jax.Array,
        target_hidden: jax.Array,
        noise_positions: jax.Array,
        ctx_positions: jax.Array,
        kv_cache_k: jax.Array,
        kv_cache_v: jax.Array,
        cache_len: jax.Array,
        actual_ctx_count: jax.Array,
        slot_idx: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Non-causal attention with on-device KV cache.

        Uses a two-phase cache write to handle padded context correctly:
          Phase A: write context K/V (with padding zeroed) at ``cache_len``.
          Phase B: write noise K/V at ``cache_len + actual_ctx_count``,
                   overwriting any padding zeros from Phase A.

        Phase 4 batched args:
            x_noise: (B, T_noise, D) noise hidden states.
            target_hidden: (B, T_padded, D) padded context features.
            noise_positions: (B, T_noise) position ids for noise tokens.
            ctx_positions: (B, T_padded) position ids for context tokens.
            kv_cache_k: (B, N_heads, max_kv_len, H) pre-allocated K cache.
            kv_cache_v: (B, N_heads, max_kv_len, H) pre-allocated V cache.
            cache_len: (B,) int, valid entries already in cache per slot.
            actual_ctx_count: (B,) int, real context tokens per slot.
            slot_idx: (B,) int, batch slot indices (identity in practice).

        Returns:
            (output, new_kv_cache_k, new_kv_cache_v)
        """
        B, T_noise, D = x_noise.shape
        T_padded = target_hidden.shape[1]

        # Project Q/K/V by flattening the batch dim so existing Einsum
        # patterns ("TD,DNH->TNH") still apply, then reshape back.
        x_noise_flat = x_noise.reshape(B * T_noise, D)
        q_flat = self.q_proj(x_noise_flat)  # (B*T_noise, N, H)
        q = q_flat.reshape(B, T_noise, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rope(
            q.reshape(B * T_noise, self.num_heads, self.head_dim),
            noise_positions.reshape(B * T_noise),
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        ).reshape(B, T_noise, self.num_heads, self.head_dim)

        # Context + noise concatenated along the token axis per slot.
        x_new = jnp.concatenate([target_hidden, x_noise], axis=1)  # (B, T_padded+T_noise, D)
        T_all = T_padded + T_noise
        x_new_flat = x_new.reshape(B * T_all, D)
        k_new_flat = self.k_proj(x_new_flat)
        v_new_flat = self.v_proj(x_new_flat)
        k_new = k_new_flat.reshape(B, T_all, self.num_kv_heads,
                                    self.head_dim)
        v_new = v_new_flat.reshape(B, T_all, self.num_kv_heads,
                                    self.head_dim)
        k_new = self.k_norm(k_new)

        new_positions = jnp.concatenate([ctx_positions, noise_positions],
                                        axis=1)  # (B, T_all)
        k_new = apply_rope(
            k_new.reshape(B * T_all, self.num_kv_heads, self.head_dim),
            new_positions.reshape(B * T_all),
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        ).reshape(B, T_all, self.num_kv_heads, self.head_dim)

        if self.num_kv_groups > 1:
            k_new = jnp.repeat(k_new, self.num_kv_groups, axis=2)
            v_new = jnp.repeat(v_new, self.num_kv_groups, axis=2)

        # Split context / noise K/V per slot along token axis.
        k_ctx = k_new[:, :T_padded]    # (B, T_padded, N, H)
        v_ctx = v_new[:, :T_padded]
        k_noise = k_new[:, T_padded:]  # (B, T_noise, N, H)
        v_noise = v_new[:, T_padded:]

        # Zero padded context entries per slot — each slot has its own
        # actual_ctx_count.
        ctx_mask = (jnp.arange(T_padded)[None, :]
                    < actual_ctx_count[:, None])  # (B, T_padded)
        ctx_mask_kv = ctx_mask[:, :, None, None]
        k_ctx = jnp.where(ctx_mask_kv, k_ctx, 0.0)
        v_ctx = jnp.where(ctx_mask_kv, v_ctx, 0.0)

        # Write per-slot K/V into the batched cache with dynamic_update_slice
        # at (slot_idx[b], 0, cache_len[b], 0). Python loop over B unrolls
        # at JIT-trace time, since B = max_num_reqs is a static shape.
        for b in range(B):
            k_ctx_b = k_ctx[b].transpose(1, 0, 2)[jnp.newaxis]  # (1,N,Tp,H)
            v_ctx_b = v_ctx[b].transpose(1, 0, 2)[jnp.newaxis]
            kv_cache_k = lax.dynamic_update_slice(
                kv_cache_k, k_ctx_b,
                (slot_idx[b], 0, cache_len[b], 0))
            kv_cache_v = lax.dynamic_update_slice(
                kv_cache_v, v_ctx_b,
                (slot_idx[b], 0, cache_len[b], 0))

        for b in range(B):
            noise_start_b = cache_len[b] + actual_ctx_count[b]
            k_noise_b = k_noise[b].transpose(1, 0, 2)[jnp.newaxis]
            v_noise_b = v_noise[b].transpose(1, 0, 2)[jnp.newaxis]
            kv_cache_k = lax.dynamic_update_slice(
                kv_cache_k, k_noise_b,
                (slot_idx[b], 0, noise_start_b, 0))
            kv_cache_v = lax.dynamic_update_slice(
                kv_cache_v, v_noise_b,
                (slot_idx[b], 0, noise_start_b, 0))

        new_cache_len = cache_len + actual_ctx_count + T_noise  # (B,)
        max_kv_len = kv_cache_k.shape[2]

        # Batched Q: (B, N, T_noise, H).
        q_4d = q.transpose(0, 2, 1, 3)
        # Batched SegmentIds. kv_ids[b,i] = 1 iff i < new_cache_len[b]; 0 else.
        kv_ids = (jnp.arange(max_kv_len)[None, :]
                  < new_cache_len[:, None]).astype(jnp.int32)  # (B, L)
        q_ids = jnp.ones((B, T_noise), dtype=jnp.int32)
        seg_ids = SegmentIds(q=q_ids, kv=kv_ids)

        sm_scale = self.head_dim_original**-0.5
        block_sizes = BlockSizes(
            block_q=T_noise,
            block_k_major=max_kv_len,
            block_k=max_kv_len,
            block_b=1,
        )
        # Batched flash_attention — leading dim matches B for Q and KV.
        attn_out = flash_attention(
            q_4d,
            kv_cache_k,
            kv_cache_v,
            segment_ids=seg_ids,
            causal=False,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            vmem_limit_bytes=_FA_VMEM_LIMIT,
        )  # (B, N, T_noise, H)

        attn_out = attn_out.transpose(0, 2, 1, 3)  # (B, T_noise, N, H)
        attn_flat = attn_out.reshape(B * T_noise, self.num_heads,
                                     self.head_dim)
        output = self.o_proj(attn_flat).reshape(B, T_noise, D)

        # DIAG: mirror of dev-rpa 0f6ff628 — print L2 norms so we can compare
        # Phase 1 per-layer activation magnitudes against dev-rpa's 30-100x
        # explosion. x=x_noise (the Q source input, matches dev-rpa semantics);
        # k/v printed over the full (ctx+noise) new K/V tensor.
        jax.debug.print(
            "[DFlashAttn] x={a} q={b} k={c} v={d} attn={e} out={f}",
            a=jnp.linalg.norm(x_noise),
            b=jnp.linalg.norm(q),
            c=jnp.linalg.norm(k_new),
            d=jnp.linalg.norm(v_new),
            e=jnp.linalg.norm(attn_out),
            f=jnp.linalg.norm(output),
        )
        return output, kv_cache_k, kv_cache_v


class DFlashMLP(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, None)),
            rngs=rng,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nnx.Module):

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = DFlashAttention(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
        )
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = DFlashMLP(config=config, dtype=dtype, rng=rng)

    def __call__(
        self,
        x: jax.Array,
        target_hidden: jax.Array,
        noise_positions: jax.Array,
        ctx_positions: jax.Array,
        kv_cache_k: jax.Array,
        kv_cache_v: jax.Array,
        cache_len: jax.Array,
        actual_ctx_count: jax.Array,
        slot_idx: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Returns (hidden_states, new_kv_cache_k, new_kv_cache_v)."""
        residual = x
        x = self.input_layernorm(x)
        x, kv_cache_k, kv_cache_v = self.self_attn(
            x,
            target_hidden,
            noise_positions,
            ctx_positions,
            kv_cache_k,
            kv_cache_v,
            cache_len,
            actual_ctx_count,
            slot_idx,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, kv_cache_k, kv_cache_v


class DFlashModel(nnx.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ) -> None:
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        hf_config = spec_config.draft_model_config.hf_config
        dtype = jnp.bfloat16
        hidden_size = hf_config.hidden_size
        rms_norm_eps = hf_config.rms_norm_eps

        self.embed_tokens = nnx.Embed(
            num_embeddings=hf_config.vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.VOCAB, None)),
            rngs=rng,
        )

        self.layers = nnx.List([
            DFlashDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
            ) for _ in range(hf_config.num_hidden_layers)
        ])

        dflash_config = getattr(hf_config, "dflash_config", {})
        target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_target_layers = getattr(hf_config, "num_target_layers", None)
        if target_layer_ids is not None:
            num_context_features = len(target_layer_ids)
        elif num_target_layers is not None:
            num_context_features = num_target_layers
        else:
            num_context_features = hf_config.num_hidden_layers

        target_hidden_size = getattr(hf_config, "target_hidden_size",
                                     hidden_size)
        fc_in_features = num_context_features * target_hidden_size

        self.fc = nnx.Linear(
            fc_in_features,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
        )

        self.hidden_norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )


class DFlashWeightLoader(BaseWeightLoader):

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh):
        super().__init__(vllm_config, framework="pt")
        self.vllm_config = vllm_config
        self.mesh = mesh

    def load_weights(self, model: "DFlashForCausalLM", mappings: dict):
        metadata_map = get_default_maps(
            self.vllm_config.speculative_config.draft_model_config,
            self.mesh,
            mappings,
        )
        load_hf_weights(
            vllm_config=self.vllm_config,
            model=model,
            metadata_map=metadata_map,
            mesh=self.mesh,
            is_draft_model=True,
        )

        # If the embedding is not initialized, initialize it with a dummy
        # array here to pass jit compilation. The real weights will be shared
        # from the target model.
        if isinstance(model.model.embed_tokens.embedding.value,
                      jax.ShapeDtypeStruct):
            model.model.embed_tokens.embedding.value = jnp.zeros(
                model.model.embed_tokens.embedding.shape,
                dtype=model.model.embed_tokens.embedding.dtype,
            )


class DFlashForCausalLM(nnx.Module):
    """DFlash draft model for speculative decoding on TPU."""

    WeightLoader = DFlashWeightLoader

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ) -> None:
        nnx.Module.__init__(self)
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        hf_config = spec_config.draft_model_config.hf_config
        self.hf_config = hf_config
        self.block_size = getattr(hf_config, "block_size", 8)
        dflash_config = getattr(hf_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get("mask_token_id", 0)

        self._position_scheme = dflash_config.get("position_scheme",
                                                  "incremental")

        self.model = DFlashModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        target_hidden_states: jax.Array,
        attention_metadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        """Forward pass for the DFlash draft model (Phase 4 batched).

        ``target_hidden_states`` is a 4-tuple:
            (ctx_hidden, cache_len_arr, actual_ctx_count_arr, slot_idx_arr)
        where:
            ctx_hidden: (B, T_padded, D) — padded context features per slot.
            cache_len_arr: (B,) int32 — valid entries in KV cache per slot.
            actual_ctx_count_arr: (B,) int32 — real context count per slot.
            slot_idx_arr: (B,) int32 — batch slot indices (identity in practice).

        ``input_ids`` is (B, T_noise): batched noise tokens per slot.

        ``kv_caches`` is a flat list of length ``2 * num_layers``:
            [k_cache_0, v_cache_0, k_cache_1, v_cache_1, ...]
        Each cache has shape ``(max_num_reqs, num_heads, max_kv_len, head_dim)``.

        Returns:
            (kv_caches, hidden_states, [target_hidden_states])
            where hidden_states has shape (B, T_noise, D).
        """
        (ctx_hidden, cache_len_arr, actual_ctx_count_arr,
         slot_idx_arr) = target_hidden_states
        cache_len = cache_len_arr           # (B,)
        actual_ctx_count = actual_ctx_count_arr  # (B,)
        slot_idx = slot_idx_arr             # (B,)

        noise_emb = self.model.embed_tokens(input_ids)  # (B, T_noise, D)
        T_padded = ctx_hidden.shape[1]
        T_noise = input_ids.shape[1]
        # Position offsets per slot: (B,) broadcast against token axes.
        if self._position_scheme == "incremental":
            pos_offset = cache_len  # (B,)
        else:
            pos_offset = jnp.zeros_like(cache_len)
        ctx_positions = (jnp.arange(T_padded, dtype=jnp.int32)[None, :]
                         + pos_offset[:, None])  # (B, T_padded)
        noise_positions = (jnp.arange(T_noise, dtype=jnp.int32)[None, :]
                           + pos_offset[:, None]
                           + actual_ctx_count[:, None])  # (B, T_noise)

        x = noise_emb
        for i, layer in enumerate(self.model.layers):
            kv_k = kv_caches[2 * i]
            kv_v = kv_caches[2 * i + 1]
            x, kv_k, kv_v = layer(
                x,
                ctx_hidden,
                noise_positions,
                ctx_positions,
                kv_k,
                kv_v,
                cache_len,
                actual_ctx_count,
                slot_idx,
            )
            kv_caches[2 * i] = kv_k
            kv_caches[2 * i + 1] = kv_v

        x = self.model.norm(x)  # (B, T_noise, D)
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Compute logits using tied embedding weights."""
        return jnp.dot(hidden_states,
                       self.model.embed_tokens.embedding.value.T)

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        """Project concatenated target auxiliary hidden states.

        Args:
            hidden_states: (T, num_target_layers * target_hidden_size)

        Returns:
            (T, hidden_size) projected + normalised context features.
        """
        return self.model.hidden_norm(self.model.fc(hidden_states))

    def load_weights(self, rng_key: jax.Array):
        self.rng = jax.random.key(self.vllm_config.model_config.seed)

        mappings = {
            "layers.*.input_layernorm": "model.layers.*.input_layernorm.scale",
            "layers.*.self_attn.q_proj":
            "model.layers.*.self_attn.q_proj.kernel",
            "layers.*.self_attn.k_proj":
            "model.layers.*.self_attn.k_proj.kernel",
            "layers.*.self_attn.v_proj":
            "model.layers.*.self_attn.v_proj.kernel",
            "layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "layers.*.self_attn.q_norm":
            "model.layers.*.self_attn.q_norm.scale",
            "layers.*.self_attn.k_norm":
            "model.layers.*.self_attn.k_norm.scale",
            "layers.*.post_attention_layernorm":
            "model.layers.*.post_attention_layernorm.scale",
            "layers.*.mlp.gate_proj": "model.layers.*.mlp.gate_proj.kernel",
            "layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.kernel",
            "layers.*.mlp.down_proj": "model.layers.*.mlp.down_proj.kernel",
            "fc": "model.fc.kernel",
            "hidden_norm": "model.hidden_norm.scale",
            "norm": "model.norm.scale",
            "embed_tokens": "model.embed_tokens.embedding",
        }

        loader = self.WeightLoader(self.vllm_config, self.mesh)
        loader.load_weights(self, mappings)

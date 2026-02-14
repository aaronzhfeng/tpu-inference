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

"""Llama DFlash draft model for speculative decoding on TPU.

Same structure as Qwen3 DFlash: projection of concatenated target hidden
states, DFlash decoder layers with non-causal attention, reuse target
embed_tokens and LM head. Adapted for Llama (LlamaConfig) and checkpoints
such as z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat.
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.dflash_attention_interface import (
    dflash_concat_attention,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.llama3 import LlamaMLP
from tpu_inference.models.jax.utils.weight_utils import (
    BaseWeightLoader,
    get_default_maps,
    load_hf_weights,
)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def _get_dflash_target_layer_ids(
    draft_hf_config: LlamaConfig,
    target_num_layers: int,
) -> List[int]:
    """Layer indices in the target model to feed into the DFlash FC input."""
    dflash_config = getattr(draft_hf_config, "dflash_config", {})
    if isinstance(dflash_config, dict):
        ids = dflash_config.get("target_layer_ids", None)
        if ids is not None:
            return list(ids)
    num_draft_layers = getattr(draft_hf_config, "num_hidden_layers", None)
    num_selected = getattr(draft_hf_config, "num_target_layers", num_draft_layers)
    if num_selected is None:
        num_selected = min(5, target_num_layers)
    if num_selected == 1:
        return [target_num_layers // 2]
    start, end = 1, target_num_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_selected - 1)))
        for i in range(num_selected)
    ]


class LlamaDFlashAttention(nnx.Module):
    """DFlash attention for Llama: Q from noise, K/V from context+noise, concat or additive."""

    def __init__(
        self,
        config: LlamaConfig,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
        quant_config: VllmQuantConfig,
        dflash_attention_impl: str,
        max_query_len: int,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps
        self.head_dim_original = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(
            self.num_heads, sharding_size
        )
        self.num_kv_heads = utils.get_padded_num_heads(
            self.num_kv_heads, sharding_size
        )
        self.mesh = mesh
        self.dflash_attention_impl = dflash_attention_impl
        if max_query_len <= 0:
            raise ValueError(f"{max_query_len=} must be positive.")
        self.max_query_len = int(max_query_len)

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.k_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = (
                utils.get_jax_dtype_from_str_dtype(kv_cache_dtype)
            )

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        target_hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        q = self.q_proj(hidden_states)
        q = self.q_norm(q)
        q = apply_rope(
            q,
            md.input_positions,
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        )
        if target_hidden_states.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "DFlash expects target/noise token counts to match, "
                f"got {target_hidden_states.shape[0]=} {hidden_states.shape[0]=}"
            )
        k_ctx = self.k_proj(target_hidden_states)
        k_ctx = self.k_norm(k_ctx)
        k_ctx = apply_rope(
            k_ctx,
            md.input_positions,
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        )
        v_ctx = self.v_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        k_noise = self.k_norm(k_noise)
        k_noise = apply_rope(
            k_noise,
            md.input_positions,
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        )
        v_noise = self.v_proj(hidden_states)
        if self.dflash_attention_impl == "additive_legacy":
            k = k_ctx + k_noise
            v = v_ctx + v_noise
            q_scale = k_scale = v_scale = None
            if self.kv_cache_quantized_dtype:
                k_scale = self._k_scale
                v_scale = self._v_scale
                k, v = quantize_kv(
                    self.kv_cache_quantized_dtype, k, v, k_scale, v_scale
                )
            new_kv_cache, outputs = attention(
                kv_cache,
                q,
                k,
                v,
                attention_metadata,
                self.mesh,
                self.head_dim_original,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        elif self.dflash_attention_impl == "concat_dense":
            outputs = dflash_concat_attention(
                q,
                k_ctx,
                k_noise,
                v_ctx,
                v_noise,
                attention_metadata,
                max_query_len=self.max_query_len,
                sm_scale=self.head_dim_original**-0.5,
            )
            q_scale = k_scale = v_scale = None
            k_for_cache = k_noise
            v_for_cache = v_noise
            if self.kv_cache_quantized_dtype:
                k_scale = self._k_scale
                v_scale = self._v_scale
                k_for_cache, v_for_cache = quantize_kv(
                    self.kv_cache_quantized_dtype,
                    k_for_cache,
                    v_for_cache,
                    k_scale,
                    v_scale,
                )
            new_kv_cache, _ = attention(
                kv_cache,
                q,
                k_for_cache,
                v_for_cache,
                attention_metadata,
                self.mesh,
                self.head_dim_original,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            raise ValueError(
                f"Unsupported {self.dflash_attention_impl=}. "
                "Expected one of {'concat_dense', 'additive_legacy'}."
            )
        o = self.o_proj(outputs)
        return new_kv_cache, o


class LlamaDFlashDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: LlamaConfig,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
        quant_config: VllmQuantConfig,
        dflash_attention_impl: str,
        max_query_len: int,
    ):
        self.input_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.self_attn = LlamaDFlashAttention(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            dflash_attention_impl=dflash_attention_impl,
            max_query_len=max_query_len,
        )
        self.post_attention_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config=config, dtype=dtype, rng=rng)

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        target_hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        kv_cache, attn_out = self.self_attn(
            kv_cache,
            hidden_states,
            target_hidden_states,
            attention_metadata,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return kv_cache, hidden_states


class LlamaDFlashModel(nnx.Module):
    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh) -> None:
        draft_model_config = vllm_config.speculative_config.draft_model_config
        hf_config = draft_model_config.hf_config
        target_model_config = vllm_config.model_config
        dtype = target_model_config.dtype
        additional_config = getattr(vllm_config, "additional_config", None) or {}

        self.embed_tokens = JaxEmbed(
            num_embeddings=target_model_config.get_vocab_size(),
            features=hf_config.hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=vllm_config.quant_config,
        )
        self.layers = [
            LlamaDFlashDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                dflash_attention_impl=additional_config.get(
                    "dflash_attention_impl", "concat_dense"
                ),
                max_query_len=int(
                    vllm_config.speculative_config.num_speculative_tokens
                )
                + 1,
            )
            for _ in range(hf_config.num_hidden_layers)
        ]
        target_layer_ids = _get_dflash_target_layer_ids(
            hf_config, target_model_config.hf_config.num_hidden_layers
        )
        self.target_layer_ids = tuple(target_layer_ids)
        target_hidden_size = target_model_config.get_hidden_size()
        combined_hidden_size = target_hidden_size * len(target_layer_ids)
        self.fc = JaxLinear(
            combined_hidden_size,
            hf_config.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=vllm_config.quant_config,
        )
        self.hidden_norm = JaxRmsNorm(
            hf_config.hidden_size,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            quant_config=vllm_config.quant_config,
        )
        self.norm = JaxRmsNorm(
            hf_config.hidden_size,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            quant_config=vllm_config.quant_config,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        target_hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        hidden_states = self.embed_tokens(input_ids)
        num_draft_layers = len(self.layers)
        draft_kv_start = max(0, len(kv_caches) - num_draft_layers)
        for i, layer in enumerate(self.layers):
            kv_idx = draft_kv_start + i
            kv_cache, hidden_states = layer(
                kv_caches[kv_idx],
                hidden_states,
                target_hidden_states,
                attention_metadata,
            )
            kv_caches[kv_idx] = kv_cache
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        return kv_caches, hidden_states, [residual]

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.fc(hidden_states)
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states


class LlamaDFlashWeightLoader(BaseWeightLoader):
    def __init__(self, vllm_config: VllmConfig, mesh: Mesh):
        super().__init__(vllm_config, framework="pt")
        self.vllm_config = vllm_config
        self.mesh = mesh

    def load_weights(self, model: "LlamaDFlashForCausalLM", mappings: dict):
        metadata_map = get_default_maps(
            self.vllm_config.speculative_config.draft_model_config,
            self.mesh,
            mappings,
        )
        filter_regex = (
            r"^((model\.)?embed_tokens\.weight|"
            r"(model\.)?layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight|"
            r"(model\.)?layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight|"
            r"(model\.)?layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight|"
            r"(model\.)?(fc|hidden_norm|norm)\.weight)$"
        )
        load_hf_weights(
            vllm_config=self.vllm_config,
            model=model,
            metadata_map=metadata_map,
            mesh=self.mesh,
            filter_regex=filter_regex,
            is_draft_model=True,
        )


class LlamaDFlashForCausalLM(nnx.Module):
    """Llama DFlash draft for speculative decoding; same interface as Qwen3 DFlash."""

    WeightLoader = LlamaDFlashWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model = LlamaDFlashModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        target_hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        return self.model(
            kv_caches,
            input_ids,
            target_hidden_states,
            attention_metadata,
        )

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.model.embed_tokens.decode(hidden_states)

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        return self.model.combine_hidden_states(hidden_states)

    def load_weights(self, _rng_key: jax.Array):
        mappings = {
            "embed_tokens": "model.embed_tokens.weight",
            "model.embed_tokens": "model.embed_tokens.weight",
            "layers.*.input_layernorm": "model.layers.*.input_layernorm.weight",
            "model.layers.*.input_layernorm": "model.layers.*.input_layernorm.weight",
            "layers.*.post_attention_layernorm": "model.layers.*.post_attention_layernorm.weight",
            "model.layers.*.post_attention_layernorm": "model.layers.*.post_attention_layernorm.weight",
            "layers.*.self_attn.q_proj": "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.q_proj": "model.layers.*.self_attn.q_proj.weight",
            "layers.*.self_attn.k_proj": "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.k_proj": "model.layers.*.self_attn.k_proj.weight",
            "layers.*.self_attn.v_proj": "model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.self_attn.v_proj": "model.layers.*.self_attn.v_proj.weight",
            "layers.*.self_attn.o_proj": "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.self_attn.o_proj": "model.layers.*.self_attn.o_proj.weight",
            "layers.*.self_attn.q_norm": "model.layers.*.self_attn.q_norm.weight",
            "model.layers.*.self_attn.q_norm": "model.layers.*.self_attn.q_norm.weight",
            "layers.*.self_attn.k_norm": "model.layers.*.self_attn.k_norm.weight",
            "model.layers.*.self_attn.k_norm": "model.layers.*.self_attn.k_norm.weight",
            "layers.*.mlp.gate_proj": "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.mlp.gate_proj": "model.layers.*.mlp.gate_proj.weight",
            "layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.weight",
            "layers.*.mlp.down_proj": "model.layers.*.mlp.down_proj.weight",
            "model.layers.*.mlp.down_proj": "model.layers.*.mlp.down_proj.weight",
            "fc": "model.fc.weight",
            "model.fc": "model.fc.weight",
            "hidden_norm": "model.hidden_norm.weight",
            "model.hidden_norm": "model.hidden_norm.weight",
            "norm": "model.norm.weight",
            "model.norm": "model.norm.weight",
        }
        loader = self.WeightLoader(self.vllm_config, self.mesh)
        loader.load_weights(self, mappings)

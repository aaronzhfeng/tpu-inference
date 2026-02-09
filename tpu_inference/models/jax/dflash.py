"""DFlash draft model implemented in JAX/Flax NNX for TPU inference.

This module implements the DFlash block-diffusion draft model for speculative
decoding on TPU, compatible with the vLLM/tpu-inference framework.

Architecture overview:
  - DFlashAttention: Cross+self attention where Q comes from noise embeddings,
    K/V comes from concatenated [context_features, noise_embeddings].
  - DFlashDecoderLayer: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual.
  - DFlashModel: Stack of decoder layers with FC projection, norms, and embedding.
  - DFlashForCausalLM: Top-level model with __call__, compute_logits,
    combine_hidden_states, and load_weights.
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen3Config
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    BaseWeightLoader,
    MetadataMap,
    get_default_maps,
    load_hf_weights,
)
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


# ---------------------------------------------------------------------------
# DFlash Attention
# ---------------------------------------------------------------------------

class DFlashAttention(nnx.Module):
    """DFlash cross+self attention.

    Q is projected from noise embeddings only.
    K/V are projected from the concatenation of [context_features, noise_embeddings]
    along the sequence dimension.  Attention is **non-causal** (full mask).
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

        self.head_dim_original = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
        self.num_heads = utils.get_padded_num_heads(self.num_heads, sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads, sharding_size)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.mesh = mesh

        # --- projections ---
        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.ATTN_HEAD, None, None)
            ),
            rngs=rng,
        )

        # QK norms (Qwen3-style)
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
        )

    def __call__(
        self,
        x_noise: jax.Array,
        target_hidden: jax.Array,
        noise_positions: jax.Array,
        ctx_positions: jax.Array,
    ) -> jax.Array:
        """
        Args:
            x_noise: (T_noise, D) noise hidden states.
            target_hidden: (T_ctx, D) projected context features.
            noise_positions: (T_noise,) position ids for noise tokens.
            ctx_positions: (T_ctx,) position ids for context tokens.

        Returns:
            output: (T_noise, D) attention output.
        """
        # Q from noise only
        q = self.q_proj(x_noise)       # (T_noise, N, H)
        q = self.q_norm(q)

        # K/V from context + noise
        x_all = jnp.concatenate([target_hidden, x_noise], axis=0)  # (S, D)
        k = self.k_proj(x_all)         # (S, K, H)
        v = self.v_proj(x_all)         # (S, K, H)
        k = self.k_norm(k)

        # RoPE
        all_positions = jnp.concatenate([ctx_positions, noise_positions], axis=0)
        q = apply_rope(
            q, noise_positions, self.head_dim_original,
            self.rope_theta, self.rope_scaling,
        )
        k = apply_rope(
            k, all_positions, self.head_dim_original,
            self.rope_theta, self.rope_scaling,
        )

        # GQA: expand K/V heads to match Q heads
        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=1)  # (S, N, H)
            v = jnp.repeat(v, self.num_kv_groups, axis=1)  # (S, N, H)

        # Non-causal attention  (q over noise, k/v over ctx+noise)
        scale = 1.0 / jnp.sqrt(jnp.float32(self.head_dim_original))
        attn_weights = jnp.einsum("tnh,snh->nts", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("nts,snh->tnh", attn_weights, v)

        return self.o_proj(attn_output)   # (T_noise, D)


# ---------------------------------------------------------------------------
# DFlash MLP  (identical to Qwen3 MLP / SwiGLU)
# ---------------------------------------------------------------------------

class DFlashMLP(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size, hidden_size, use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, None)),
            rngs=rng,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# DFlash Decoder Layer
# ---------------------------------------------------------------------------

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
            hidden_size, epsilon=rms_norm_eps, param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)), rngs=rng,
        )
        self.self_attn = DFlashAttention(
            config=config, dtype=dtype, rng=rng, mesh=mesh,
        )
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size, epsilon=rms_norm_eps, param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)), rngs=rng,
        )
        self.mlp = DFlashMLP(config=config, dtype=dtype, rng=rng)

    def __call__(
        self,
        x: jax.Array,
        target_hidden: jax.Array,
        noise_positions: jax.Array,
        ctx_positions: jax.Array,
    ) -> jax.Array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, target_hidden, noise_positions, ctx_positions)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# DFlash Inner Model
# ---------------------------------------------------------------------------

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

        # Embedding (will be shared from target model)
        self.embed_tokens = nnx.Embed(
            num_embeddings=hf_config.vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.VOCAB, None)),
            rngs=rng,
        )

        # Decoder layers
        self.layers = [
            DFlashDecoderLayer(
                config=hf_config, dtype=dtype, rng=rng, mesh=mesh,
            )
            for _ in range(hf_config.num_hidden_layers)
        ]

        # FC: projects concatenated target hidden states to hidden_size
        dflash_config = getattr(hf_config, "dflash_config", {})
        target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_target_layers = getattr(hf_config, "num_target_layers", None)
        if target_layer_ids is not None:
            num_context_features = len(target_layer_ids)
        elif num_target_layers is not None:
            num_context_features = num_target_layers
        else:
            num_context_features = hf_config.num_hidden_layers

        # Target model hidden size may differ from draft; default to same.
        target_hidden_size = getattr(hf_config, "target_hidden_size", hidden_size)
        fc_in_features = num_context_features * target_hidden_size

        self.fc = nnx.Linear(
            fc_in_features, hidden_size, use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
        )

        self.hidden_norm = nnx.RMSNorm(
            hidden_size, epsilon=rms_norm_eps, param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)), rngs=rng,
        )
        self.norm = nnx.RMSNorm(
            hidden_size, epsilon=rms_norm_eps, param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)), rngs=rng,
        )


# ---------------------------------------------------------------------------
# DFlash Weight Loader
# ---------------------------------------------------------------------------

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

        # If embedding is not yet initialized, fill with zeros.
        # Real embedding weights will be shared from the target model.
        if isinstance(
            model.model.embed_tokens.embedding.value, jax.ShapeDtypeStruct
        ):
            model.model.embed_tokens.embedding.value = jnp.zeros(
                model.model.embed_tokens.embedding.shape,
                dtype=model.model.embed_tokens.embedding.dtype,
            )


# ---------------------------------------------------------------------------
# DFlash Top-Level Model
# ---------------------------------------------------------------------------

class DFlashForCausalLM(nnx.Module):
    """DFlash draft model for speculative decoding on TPU.

    Compatible with the vLLM / tpu-inference model registry interface.
    """

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

        self.model = DFlashModel(
            vllm_config=vllm_config, rng=self.rng, mesh=mesh,
        )

    # ----- forward pass ---------------------------------------------------

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        target_hidden_states: jax.Array,
        attention_metadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        """Forward pass for the DFlash draft model.

        Args:
            kv_caches: KV cache arrays (passed through unchanged).
            input_ids: (T_noise,) token IDs for the draft block.
            target_hidden_states: (T_ctx, D) already-projected context features
                (output of combine_hidden_states).
            attention_metadata: AttentionMetadata with input_positions, etc.

        Returns:
            Tuple of (kv_caches, hidden_states, [target_hidden_states]).
        """
        # Embed the draft block tokens
        noise_emb = self.model.embed_tokens(input_ids)   # (T_noise, D)

        # Derive positions
        noise_positions = attention_metadata.input_positions  # (T_noise,)
        ctx_len = target_hidden_states.shape[0]
        first_noise_pos = noise_positions[0]
        ctx_positions = jnp.arange(ctx_len) + jnp.maximum(
            first_noise_pos - ctx_len, 0
        )

        # Run through decoder layers
        x = noise_emb
        for layer in self.model.layers:
            x = layer(x, target_hidden_states, noise_positions, ctx_positions)

        x = self.model.norm(x)

        # Return target_hidden_states as "residual" so that the proposer
        # can pass them back in subsequent iterations.
        return kv_caches, x, [target_hidden_states]

    # ----- logits ---------------------------------------------------------

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Compute logits using tied embedding weights."""
        return jnp.dot(
            hidden_states, self.model.embed_tokens.embedding.value.T
        )

    # ----- combine hidden states ------------------------------------------

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        """Project concatenated target auxiliary hidden states.

        Args:
            hidden_states: (T, num_target_layers * target_hidden_size)

        Returns:
            (T, hidden_size) projected + normalised context features.
        """
        return self.model.hidden_norm(self.model.fc(hidden_states))

    # ----- weight loading -------------------------------------------------

    def load_weights(self, rng_key: jax.Array):
        self.rng = jax.random.key(self.vllm_config.model_config.seed)

        mappings = {
            # Decoder-layer weights (wildcard over layer index)
            "layers.*.input_layernorm":
                "model.layers.*.input_layernorm.scale",
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
            "layers.*.mlp.gate_proj":
                "model.layers.*.mlp.gate_proj.kernel",
            "layers.*.mlp.up_proj":
                "model.layers.*.mlp.up_proj.kernel",
            "layers.*.mlp.down_proj":
                "model.layers.*.mlp.down_proj.kernel",
            # Top-level weights
            "fc": "model.fc.kernel",
            "hidden_norm": "model.hidden_norm.scale",
            "norm": "model.norm.scale",
            "embed_tokens": "model.embed_tokens.embedding",
        }

        loader = self.WeightLoader(self.vllm_config, self.mesh)
        loader.load_weights(self, mappings)

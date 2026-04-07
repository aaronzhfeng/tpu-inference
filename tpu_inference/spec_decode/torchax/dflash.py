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
"""DFlash proposer for speculative decoding via torchax.

Loads the original HuggingFace DFlash model through the torchax wrapper
and runs stateless (no per-layer KV cache) draft iterations.  Each
``propose`` call processes the full accumulated context via non-causal
attention, avoiding the complexity of on-device KV cache management at
the cost of recomputing K/V projections each iteration.

This provides a torchax-compatible DFlash path for the TPU Inference
production stack, complementing the JAX-native implementation in
``spec_decode.jax.dflash``.
"""

import functools
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.torchax.dflash_torchax import DFlashTorchaxWrapper
from tpu_inference.utils import device_array

logger = init_logger(__name__)


class DFlashTorchaxProposer:
    """Proposer for speculative decoding using DFlash via torchax.

    Wraps the HuggingFace DFlash PyTorch model through torchax for TPU
    execution.  Runs stateless (no KV caching) — each draft iteration
    processes the full accumulated context.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        runner: Any,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.mesh = runner.mesh
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens
        )

        hf_config = self.draft_model_config.hf_config
        self.block_size = getattr(
            hf_config, "block_size", self.num_speculative_tokens + 1
        )
        dflash_config = getattr(hf_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get("mask_token_id", 0)
        self.hidden_size = hf_config.hidden_size
        self.num_layers = hf_config.num_hidden_layers

        # Determine raw hidden dimension (num_target_layers * target_hidden_size)
        target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_target_layers = getattr(hf_config, "num_target_layers", None)
        if target_layer_ids is not None:
            self._num_target_layers = len(target_layer_ids)
        elif num_target_layers is not None:
            self._num_target_layers = num_target_layers
        else:
            self._num_target_layers = hf_config.num_hidden_layers
        target_hidden_size = getattr(
            hf_config, "target_hidden_size", self.hidden_size)
        self._raw_hidden_dim = self._num_target_layers * target_hidden_size

        self.rng_key = jax.random.key(self.vllm_config.model_config.seed)
        self.max_num_tokens = runner.max_num_tokens
        self.max_model_len = runner.max_model_len

        # Context length tracking
        self._ctx_len: int = 0
        self._prev_seq_len: int = 0

        # On-device context buffer (allocated in load_model)
        self._ctx_buf: Optional[jax.Array] = None

        # Torchax wrapper (allocated in load_model)
        self._wrapper: Optional[DFlashTorchaxWrapper] = None
        self._draft_forward_fn = None
        self._compute_logits_fn = None
        self._params: Optional[dict] = None
        self._embed_weight: Optional[jax.Array] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, target_model: Any) -> None:
        """Load the DFlash draft model via torchax and share embeddings."""
        self._wrapper = DFlashTorchaxWrapper(self.mesh)

        # Resolve draft model path
        draft_model_path = self.draft_model_config.model
        self._wrapper.load(draft_model_path, target_model)

        # Get JIT-compiled functions
        self._draft_forward_fn = self._wrapper.get_draft_forward_fn()
        self._compute_logits_fn = self._wrapper.get_compute_logits_fn()
        self._params = self._wrapper.params
        self._embed_weight = self._wrapper.embed_weight_jax

        # Allocate on-device context buffer
        buf_len = self._next_padded_size(self.max_model_len)
        self._ctx_buf = jnp.zeros(
            (buf_len, self._raw_hidden_dim), dtype=jnp.bfloat16)

        logger.info(
            "DFlash torchax proposer loaded: context buffer shape %s",
            self._ctx_buf.shape,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _next_padded_size(n: int) -> int:
        """Round n up to the next power-of-two (min 16)."""
        if n <= 16:
            return 16
        p = 16
        while p < n:
            p *= 2
        return p

    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _build_noise_block(
        self,
        seq_len_arr: jax.Array,
        next_token_ids: jax.Array,
        mask_token_id: int,
        block_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Build noise block and positions (JIT-compiled)."""
        seq_len = seq_len_arr[0]
        first_token = next_token_ids[0]
        noise_input_ids = jnp.full(
            (block_size,), mask_token_id, dtype=jnp.int32
        )
        noise_input_ids = noise_input_ids.at[0].set(first_token)
        noise_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        return noise_input_ids, noise_positions

    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample_block_draft_tokens(
        self,
        params: dict,
        hidden_states: jax.Array,
        embed_weight: jax.Array,
    ) -> jax.Array:
        """Greedy-sample draft tokens from the block output."""
        draft_hidden = hidden_states[1 : 1 + self.num_speculative_tokens]
        logits = self._compute_logits_fn(params, draft_hidden, embed_weight)
        return jnp.argmax(logits, axis=-1)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        next_token_ids: jax.Array,
        num_rejected_tokens: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare DFlash inputs for stateless torchax forward."""
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0

        # 1. Current sequence length
        seq_len_jax = attn_metadata.seq_lens[0]
        seq_len = int(jax.device_get(seq_len_jax))

        # 2. Crop semantics: match GPU DynamicCache.crop(start).
        if self._prev_seq_len > 0 and seq_len < self._ctx_len:
            self._ctx_len = seq_len
        self._prev_seq_len = seq_len

        # 3. Concatenate raw target hidden states (on-device)
        raw_hidden = jnp.concatenate(aux_hidden_states, axis=-1)

        # 4. Update on-device context buffer
        num_new = seq_len - self._ctx_len
        if num_new <= 0:
            # Rejection — trim context tracking
            self._ctx_len = seq_len
        else:
            end = min(self._ctx_len + num_new, self.max_model_len)
            n_copy = end - self._ctx_len
            # Write new tokens into buffer (on-device, no host transfer)
            new_raw = raw_hidden[:n_copy].astype(jnp.bfloat16)
            self._ctx_buf = lax.dynamic_update_slice(
                self._ctx_buf, new_raw, (self._ctx_len, 0)
            )
            self._ctx_len = end

        # 5. Pad context to power-of-2 and build positions + mask.
        padded_ctx = self._next_padded_size(max(self._ctx_len, 1))
        total_len = padded_ctx + self.block_size

        # Slice the relevant context portion from the buffer.
        # padded_ctx is a Python int (power-of-2), so this is a static slice.
        ctx_padded = self._ctx_buf[:padded_ctx]

        # Position IDs: context positions [0, 1, ..., ctx_len-1, 0, 0, ...],
        # then noise positions [ctx_len, ctx_len+1, ..., ctx_len+block_size-1]
        ctx_positions = jnp.where(
            jnp.arange(padded_ctx) < self._ctx_len,
            jnp.arange(padded_ctx, dtype=jnp.int32),
            jnp.zeros(padded_ctx, dtype=jnp.int32),
        )
        noise_positions = (
            jnp.arange(self.block_size, dtype=jnp.int32) + self._ctx_len
        )
        position_ids = jnp.concatenate([ctx_positions, noise_positions])

        # Attention mask: 1 for valid context + all noise, 0 for padding
        ctx_mask = (jnp.arange(padded_ctx) < self._ctx_len).astype(jnp.int32)
        noise_mask = jnp.ones(self.block_size, dtype=jnp.int32)
        attention_mask = jnp.concatenate([ctx_mask, noise_mask])

        # Pack as target_hidden_states (opaque to the manager)
        target_hidden_states = (ctx_padded, position_ids, attention_mask)

        # 6. Build noise block
        seq_len_arr = device_array(
            self.mesh, np.array([seq_len], dtype=np.int32)
        )
        noise_input_ids, _ = self._build_noise_block(
            seq_len_arr,
            next_token_ids,
            self.mask_token_id,
            self.block_size,
        )

        # 7. Build draft attention metadata
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id]
            .get_cpu_tensor()
            .reshape(-1)
        )
        num_reqs = attn_metadata.seq_lens.shape[0]
        draft_attn_metadata = replace(
            attn_metadata,
            input_positions=noise_positions,
            query_start_loc=jnp.array(
                [0, self.block_size], dtype=jnp.int32
            ),
            block_tables=device_array(self.mesh, block_tables),
        )

        dummy_last_indices = jnp.zeros(num_reqs, dtype=jnp.int32)
        return (
            target_hidden_states,
            noise_input_ids,
            dummy_last_indices,
            draft_attn_metadata,
        )

    # ------------------------------------------------------------------
    # Draft token generation
    # ------------------------------------------------------------------

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Generate all draft tokens in one forward pass via torchax."""
        ctx_padded, position_ids, attention_mask = target_hidden_states

        # Run the HF DFlash model through torchax (stateless, no KV cache)
        hidden_states = self._draft_forward_fn(
            self._params,
            input_ids,
            ctx_padded,
            position_ids,
            self._embed_weight,
            attention_mask,
        )

        # Sample draft tokens
        draft_token_ids = self._sample_block_draft_tokens(
            self._params, hidden_states, self._embed_weight
        )

        if draft_token_ids.ndim == 1:
            draft_token_ids = draft_token_ids[jnp.newaxis, :]

        # Pass the framework kv_caches through unchanged
        return kv_caches, draft_token_ids

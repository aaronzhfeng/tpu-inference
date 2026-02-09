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
"""DFlash proposer for speculative decoding on JAX/TPU.

Uses **torchax** to run the original PyTorch DFlash model on TPU,
guaranteeing numerical equivalence with the GPU reference.

DFlash is a block-diffusion draft model that produces an entire block of
draft tokens in a single forward pass:

    noise_block  = [last_accepted_token, mask, mask, …, mask]  (block_size)
    hidden       = draft_model(noise_block, context_hidden)    (block_size, D)
    draft_ids    = argmax(lm_head(hidden[1:]))                 (block_size-1)
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


class DFlashProposer:
    """Proposer for speculative decoding using the DFlash block-diffusion
    draft model, loaded via torchax."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        runner: Any,  # TPUModelRunner
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
        self.target_layer_ids = dflash_config.get("target_layer_ids", None)

        self.rng_key = jax.random.key(self.vllm_config.model_config.seed)
        self.max_num_tokens = runner.max_num_tokens

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, target_model: Any) -> None:
        """Load the PyTorch DFlash model via torchax."""
        draft_model_path = self.draft_model_config.model

        self.wrapper = DFlashTorchaxWrapper(self.mesh)
        self.wrapper.load(draft_model_path, target_model)

        # Build JIT-compiled callables
        self._draft_forward_fn = self.wrapper.get_draft_forward_fn()
        self._combine_fn = self.wrapper.get_combine_hidden_fn()
        self._logits_fn = self.wrapper.get_compute_logits_fn()

        self.params = self.wrapper.params
        self.embed_weight = self.wrapper.embed_weight_jax

        logger.info("DFlash proposer loaded (torchax).")

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0, 4, 5))
    def _build_noise_block_and_context(
        self,
        aux_hidden_states: tuple[jax.Array, ...],
        seq_lens: jax.Array,
        next_token_ids: jax.Array,
        mask_token_id: int,
        block_size: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Build noise block, combine raw aux hidden states, derive positions.

        Returns:
            raw_target_hidden: (T_ctx, num_layers * D) concatenated (NOT
                               projected – projection happens inside the
                               PyTorch model via fc + hidden_norm).
            noise_input_ids:   (block_size,) [next_token, mask, …, mask].
            position_ids:      (T_ctx + block_size,) for RoPE.
        """
        # Concatenate aux hidden states from different target layers
        raw_target_hidden = jnp.concatenate(aux_hidden_states, axis=-1)

        # Noise block
        seq_len = seq_lens[0]
        first_token = next_token_ids[0]
        noise_input_ids = jnp.full((block_size,), mask_token_id,
                                   dtype=jnp.int32)
        noise_input_ids = noise_input_ids.at[0].set(first_token)

        # Position IDs: [0 .. ctx_len-1, ctx_len .. ctx_len+block_size-1].
        # Use int32 – JAX on TPU truncates int64 anyway.
        ctx_len = raw_target_hidden.shape[0]
        total_len = ctx_len + block_size
        position_ids = jnp.arange(total_len, dtype=jnp.int32)

        # Zero out padding entries (>= seq_len) so they contribute ~zero
        # to the attention computation.
        ctx_mask = (jnp.arange(ctx_len) < seq_len).astype(
            raw_target_hidden.dtype)
        raw_target_hidden = raw_target_hidden * ctx_mask[:, None]

        return raw_target_hidden, noise_input_ids, position_ids

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        next_token_ids: jax.Array,
        num_rejected_tokens: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare DFlash-specific inputs.

        Returns a 4-tuple for API compatibility with the Eagle3 flow:
            (raw_target_hidden, noise_input_ids, position_ids, attn_metadata)
        The 3rd element is position_ids (not last_token_indices).
        """
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id]
            .get_cpu_tensor()
            .reshape(-1)
        )

        raw_target_hidden, noise_input_ids, position_ids = (
            self._build_noise_block_and_context(
                aux_hidden_states,
                attn_metadata.seq_lens,
                next_token_ids,
                self.mask_token_id,
                self.block_size,
            )
        )

        draft_attn_metadata = replace(
            attn_metadata,
            block_tables=device_array(self.mesh, block_tables),
        )

        return (
            raw_target_hidden,
            noise_input_ids,
            position_ids,
            draft_attn_metadata,
        )

    # ------------------------------------------------------------------
    # Draft token generation
    # ------------------------------------------------------------------

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,       # noise_input_ids from prepare_inputs
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,  # actually position_ids
        target_hidden_states: jax.Array,  # raw_target_hidden
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Generate all draft tokens in one forward pass via torchax.

        Returns:
            (kv_caches, draft_token_ids)  where draft_token_ids has shape
            (1, num_speculative_tokens).
        """
        position_ids = last_token_indices  # repurposed from API slot

        # Run the PyTorch DFlash model through torchax.
        hidden_states = self._draft_forward_fn(
            self.params,
            input_ids,             # (block_size,)
            target_hidden_states,  # (ctx_len, num_layers * D)
            position_ids,          # (ctx_len + block_size,)
            self.embed_weight,     # (vocab_size, D)
        )
        # hidden_states: (block_size, D)

        # Take positions 1..block_size-1 → num_speculative_tokens predictions
        draft_hidden = hidden_states[1: 1 + self.num_speculative_tokens]

        # compute logits via tied embeddings
        logits = self._logits_fn(
            self.params,
            draft_hidden,
            self.embed_weight,
        )

        draft_token_ids = jnp.argmax(logits, axis=-1)
        draft_token_ids = lax.with_sharding_constraint(
            draft_token_ids, NamedSharding(self.mesh, PartitionSpec()))

        # Framework expects (num_reqs, num_speculative_tokens)
        if draft_token_ids.ndim == 1:
            draft_token_ids = draft_token_ids[jnp.newaxis, :]

        return kv_caches, draft_token_ids

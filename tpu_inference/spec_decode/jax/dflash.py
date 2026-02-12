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

    def _build_noise_block_and_context(
        self,
        aux_hidden_states: tuple[jax.Array, ...],
        next_token_ids: jax.Array,
        seq_len: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Build the DFlash context/noise inputs for one active request.

        Returns:
            raw_target_hidden: (T_ctx, num_layers * D) concatenated (NOT
                               projected – projection happens inside the
                               PyTorch model via fc + hidden_norm).
            noise_input_ids:   (block_size,) [next_token, mask, …, mask].
            position_ids:      (T_ctx + block_size,) for RoPE.
        """
        # Trim to real (non-padded) context length before concatenation.
        trimmed_hidden = [h[:seq_len] for h in aux_hidden_states]
        raw_target_hidden = jnp.concatenate(trimmed_hidden, axis=-1)

        # Noise block
        first_token = next_token_ids[0]
        noise_input_ids = jnp.full((self.block_size,), self.mask_token_id,
                                   dtype=jnp.int32)
        noise_input_ids = noise_input_ids.at[0].set(first_token)

        # Position IDs: [0 .. ctx_len-1, ctx_len .. ctx_len+block_size-1].
        total_len = int(seq_len) + int(self.block_size)
        position_ids = jnp.arange(total_len, dtype=jnp.int32)

        return raw_target_hidden, noise_input_ids, position_ids

    def _resolve_context_seq_len(
        self,
        attn_metadata: AttentionMetadata,
        aux_hidden_states: tuple[jax.Array, ...],
        num_rejected_tokens: Optional[jax.Array],
    ) -> int:
        """Resolve true context length for DFlash input construction.

        The runner metadata for speculative steps still reflects scheduled
        query tokens before rejection. For DFlash, we must trim rejected tail
        tokens so the next draft sees only accepted+bonus context.
        """
        seq_lens_cpu = np.asarray(jax.device_get(attn_metadata.seq_lens)).astype(
            np.int32
        )
        if seq_lens_cpu.size == 0:
            raise RuntimeError("DFlash expected non-empty seq_lens metadata.")

        seq_len = int(seq_lens_cpu[0])
        if num_rejected_tokens is not None:
            rejected_cpu = np.asarray(jax.device_get(num_rejected_tokens)).astype(
                np.int32
            )
            if rejected_cpu.size > 0:
                seq_len -= int(rejected_cpu[0])

        # Keep at least one token (the current accepted token).
        seq_len = max(1, seq_len)
        # Defensive clamp to actual hidden-state buffer length.
        max_ctx = min(int(h.shape[0]) for h in aux_hidden_states)
        return min(seq_len, max_ctx)

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
        if self.runner.input_batch.num_reqs != 1:
            raise NotImplementedError(
                "DFlash torchax proposer currently supports max_num_seqs=1."
            )

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id]
            .get_cpu_tensor()
            .reshape(-1)
        )

        seq_len = self._resolve_context_seq_len(
            attn_metadata,
            aux_hidden_states,
            num_rejected_tokens,
        )
        raw_target_hidden, noise_input_ids, position_ids = (
            self._build_noise_block_and_context(
                aux_hidden_states,
                next_token_ids,
                seq_len,
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
    ) -> tuple[list[jax.Array], jnp.ndarray, jnp.ndarray]:
        """Generate all draft tokens in one forward pass via torchax.

        Returns:
            (kv_caches, draft_token_ids, draft_token_probs) where
            `draft_token_ids` has shape (1, num_speculative_tokens) and
            `draft_token_probs` has shape (1, num_speculative_tokens).
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
        draft_log_probs = jax.nn.log_softmax(logits.astype(jnp.float32),
                                             axis=-1)
        draft_token_probs = jnp.take_along_axis(
            draft_log_probs, draft_token_ids[:, None], axis=-1).squeeze(-1)
        draft_token_probs = jnp.exp(draft_token_probs)
        draft_token_ids = lax.with_sharding_constraint(
            draft_token_ids, NamedSharding(self.mesh, PartitionSpec()))
        draft_token_probs = lax.with_sharding_constraint(
            draft_token_probs, NamedSharding(self.mesh, PartitionSpec()))

        # Framework expects (num_reqs, num_speculative_tokens)
        if draft_token_ids.ndim == 1:
            draft_token_ids = draft_token_ids[jnp.newaxis, :]
        if draft_token_probs.ndim == 1:
            draft_token_probs = draft_token_probs[jnp.newaxis, :]

        return kv_caches, draft_token_ids, draft_token_probs

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
"""DFlash proposer for speculative decoding on JAX/TPU."""

import functools
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.utils import device_array, get_mesh_shape_product

logger = init_logger(__name__)


class DFlashProposer:
    """Proposer for speculative decoding using DFlash block diffusion."""

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
            self.speculative_config.num_speculative_tokens)

        hf_config = self.draft_model_config.hf_config
        self.block_size = getattr(hf_config, "block_size",
                                  self.num_speculative_tokens + 1)
        dflash_config = getattr(hf_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get("mask_token_id", 0)
        self.hidden_size = hf_config.hidden_size
        self.num_layers = hf_config.num_hidden_layers

        self.rng_key = jax.random.key(self.vllm_config.model_config.seed)
        self.max_num_tokens = runner.max_num_tokens
        self.max_model_len = runner.max_model_len
        self._max_num_reqs = runner.max_num_reqs

        # Per-slot state (slot = batch position in vLLM's scheduler, 0..max_num_reqs-1)
        # When a slot is reassigned to a new request_id, state for that slot
        # is reset. This is Phase 1 batched-support groundwork: at num_reqs=1
        # only slot 0 is ever used, preserving existing single-request behavior.
        #
        # - _ctx_len_slot: host-side counter of accepted context length.
        # - _cache_len_slot: position in the draft KV cache where the next
        #   noise block is written.
        # - _prev_seq_len_slot: seq_len from the PREVIOUS prepare_inputs call.
        #   GPU reference calls past_key_values_draft.crop(start) AFTER each
        #   forward pass, where start = beginning of the CURRENT iteration's
        #   block == seq_len from the PREVIOUS call. We match this by
        #   restoring cache_len = prev_seq_len at the start of each step.
        self._ctx_len_slot = np.zeros(self._max_num_reqs, dtype=np.int64)
        self._cache_len_slot = np.zeros(self._max_num_reqs, dtype=np.int64)
        self._prev_seq_len_slot = np.zeros(self._max_num_reqs, dtype=np.int64)
        self._slot_req_id: list = [None] * self._max_num_reqs

        # On-device KV caches (allocated in load_model)
        self._draft_kv_caches: Optional[list[jax.Array]] = None
        self._max_kv_len: int = 0

    def load_model(self, target_model: Any) -> None:
        """Load the DFlash draft model and share embeddings from target."""
        (
            self.model_fn,
            _,  # draft compute_logits — NOT used; DFlash uses target LM head
            _,
            self.combine_hidden_states_fn,
            _,
            self.state,
            _,
            _,
        ) = get_model(self.vllm_config,
                      self.rng_key,
                      self.mesh,
                      is_draft_model=True)

        # DFlash contract: draft hidden states are decoded with the TARGET
        # model's LM head (shared embedding weights), not the draft's own.
        self.target_compute_logits_fn = self.runner.compute_logits_fn
        self.target_state = self.runner.state

        # Share the target model's embedding with the draft model.
        # IMPORTANT: Copy embedding VALUES only, not the entire module.
        # The target uses JaxEmbed (stores param as .weight) while the draft
        # uses nnx.Embed (stores param as .embedding). Replacing the entire
        # module breaks the graphdef-state pytree alignment, causing the
        # JIT'd forward to use corrupted embeddings.
        draft_embed = getattr(self.state.model, "embed_tokens", None)
        target_embed = getattr(target_model.model, "embed_tokens", None)
        if target_embed is None:
            target_embed = getattr(target_model.model, "embed", None)
        if target_embed is not None and draft_embed is not None:
            # Debug: log available attributes
            logger.info(f"Draft embed type: {type(draft_embed)}, "
                        f"attrs: {list(draft_embed.__dict__.keys()) if hasattr(draft_embed, '__dict__') else dir(draft_embed)[:10]}")
            logger.info(f"Target embed type: {type(target_embed)}, "
                        f"attrs: {list(target_embed.__dict__.keys()) if hasattr(target_embed, '__dict__') else dir(target_embed)[:10]}")
            # Get embedding parameter - handle both nnx.Embed (.embedding) and JaxEmbed (.weight)
            draft_embed_param = getattr(draft_embed, 'embedding', None) or getattr(draft_embed, 'weight', None)
            target_embed_param = getattr(target_embed, 'embedding', None) or getattr(target_embed, 'weight', None)
            if draft_embed_param is None or target_embed_param is None:
                logger.warning(f"Cannot find embedding param: draft={draft_embed_param}, target={target_embed_param}")
            else:
                target_embed_value = target_embed_param.value
                draft_embed_value = draft_embed_param.value
            if not jnp.any(draft_embed_value):
                logger.info(
                    "Sharing target model embedding with DFlash draft model.")
                draft_embed_param.value = target_embed_value
            elif jnp.array_equal(draft_embed_value, target_embed_value):
                logger.info("Draft embedding identical to target; sharing.")
                draft_embed_param.value = target_embed_value

        # Allocate on-device KV caches
        hf_config = self.draft_model_config.hf_config

        sharding_size = get_mesh_shape_product(self.mesh,
                                               ShardingAxisName.MLP_TENSOR)
        num_heads = utils.get_padded_num_heads(hf_config.num_attention_heads,
                                               sharding_size)
        head_dim_orig = getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads)
        head_dim = utils.get_padded_head_dim(head_dim_orig)

        self._max_kv_len = self._next_padded_size(self.max_model_len)
        # Phase 2: batch dim = max_num_reqs so each scheduler slot has its
        # own KV cache row. Phase 2 only sizes the cache; writes/reads still
        # target slot 0 until Phase 3 routes per-slot. At num_reqs=1 only
        # slot 0 is touched, preserving existing behavior.
        cache_shape = (self._max_num_reqs, num_heads, self._max_kv_len,
                       head_dim)
        self._draft_kv_caches = []
        for _ in range(self.num_layers):
            k_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
            v_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
            self._draft_kv_caches.append(k_cache)
            self._draft_kv_caches.append(v_cache)

        logger.info(
            "Allocated DFlash on-device KV caches: %d layers, shape %s",
            self.num_layers,
            cache_shape,
        )

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _project_aux_hidden(
            self, state: nnx.State,
            aux_hidden_states: tuple[jax.Array, ...]) -> jax.Array:
        """Project and normalise auxiliary hidden states."""
        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        return self.combine_hidden_states_fn(state, raw)

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
        noise_input_ids = jnp.full((block_size, ),
                                   mask_token_id,
                                   dtype=jnp.int32)
        noise_input_ids = noise_input_ids.at[0].set(first_token)
        noise_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        return noise_input_ids, noise_positions

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        next_token_ids: jax.Array,
        num_rejected_tokens: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare DFlash inputs with on-device KV cache.

        Phase 4 batched: all per-slot data is stacked to leading dim
        B = self._max_num_reqs so the JIT'd model forward is shape-stable
        regardless of num_reqs. Inactive slots (idx >= num_reqs) carry
        zero cache_len and zero actual_ctx_count, so their attention
        output is discarded by the manager (only [:num_reqs] is read).
        """
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0

        num_reqs = attn_metadata.seq_lens.shape[0]
        B = self._max_num_reqs

        # Read per-slot metadata from CPU-side attrs (avoid device_get sync).
        if hasattr(attn_metadata, '_cpu_seq_lens'):
            seq_lens_cpu = attn_metadata._cpu_seq_lens
        else:
            seq_lens_cpu = jax.device_get(attn_metadata.seq_lens)
        if hasattr(attn_metadata, '_cpu_query_start_loc'):
            qsl_cpu = attn_metadata._cpu_query_start_loc
        else:
            qsl_cpu = jax.device_get(attn_metadata.query_start_loc)

        current_req_ids = self.runner.input_batch.req_ids[:num_reqs]

        # 1. Slot state updates + per-slot n_copy compute (host side).
        # Per-slot counters mirror the single-slot logic: reset on req_id
        # change, crop cache via prev_seq_len, bound n_copy by slot's tokens
        # this step and by max_model_len.
        per_slot_cache_len = np.zeros(B, dtype=np.int32)
        per_slot_n_copy = np.zeros(B, dtype=np.int32)
        per_slot_seq_len = np.zeros(B, dtype=np.int32)
        per_slot_token_start = np.zeros(B, dtype=np.int32)
        for slot in range(num_reqs):
            if self._slot_req_id[slot] != current_req_ids[slot]:
                self._ctx_len_slot[slot] = 0
                self._cache_len_slot[slot] = 0
                self._prev_seq_len_slot[slot] = 0
                self._slot_req_id[slot] = current_req_ids[slot]
            seq_len = int(seq_lens_cpu[slot])
            # Crop cache to match GPU DynamicCache.crop(start) semantics —
            # see the docstring on _cache_len_slot in __init__.
            if self._prev_seq_len_slot[slot] > 0:
                self._cache_len_slot[slot] = self._prev_seq_len_slot[slot]
            if seq_len < self._ctx_len_slot[slot]:
                self._ctx_len_slot[slot] = seq_len
            self._prev_seq_len_slot[slot] = seq_len
            num_new = seq_len - int(self._ctx_len_slot[slot])
            slot_tok_start = int(qsl_cpu[slot])
            slot_tok_end = int(qsl_cpu[slot + 1])
            slot_tokens = slot_tok_end - slot_tok_start
            if num_new <= 0:
                self._ctx_len_slot[slot] = seq_len
                self._cache_len_slot[slot] = min(
                    int(self._cache_len_slot[slot]), seq_len)
                n_copy = 0
            else:
                end = min(int(self._ctx_len_slot[slot]) + num_new,
                          self.max_model_len)
                n_copy = min(end - int(self._ctx_len_slot[slot]),
                             slot_tokens)
                self._ctx_len_slot[slot] = (int(self._ctx_len_slot[slot])
                                            + n_copy)
            per_slot_cache_len[slot] = int(self._cache_len_slot[slot])
            per_slot_n_copy[slot] = n_copy
            per_slot_seq_len[slot] = seq_len
            per_slot_token_start[slot] = slot_tok_start

        # 2. Project aux hidden states once (JIT'd, batch-oblivious).
        projected = self._project_aux_hidden(self.state, aux_hidden_states)

        # 3. Batched context tensor: (B, common_padded, D). Power-of-2 padded
        # to the max n_copy across active slots so JIT only traces a small
        # number of shapes (same set as the Phase 3 single-slot padding).
        max_n_copy = int(per_slot_n_copy.max()) if num_reqs > 0 else 0
        common_padded = self._next_padded_size(max(max_n_copy, 1))
        ctx_parts = []
        for slot in range(B):
            n_copy = int(per_slot_n_copy[slot]) if slot < num_reqs else 0
            if n_copy > 0:
                tok_start = int(per_slot_token_start[slot])
                slot_ctx = projected[tok_start:tok_start + n_copy].astype(
                    jnp.bfloat16)
                pad_size = common_padded - n_copy
                if pad_size > 0:
                    slot_ctx = jnp.concatenate([
                        slot_ctx,
                        jnp.zeros((pad_size, self.hidden_size),
                                  dtype=jnp.bfloat16),
                    ], axis=0)
                ctx_parts.append(slot_ctx[jnp.newaxis, :, :])
            else:
                ctx_parts.append(
                    jnp.zeros((1, common_padded, self.hidden_size),
                              dtype=jnp.bfloat16))
        ctx_batched = jnp.concatenate(ctx_parts, axis=0)
        ctx_batched = device_array(self.mesh, ctx_batched)

        # 4. Batched noise block: (B, block_size) for input_ids + positions.
        # Slot b's first noise token = next_token_ids[b]; rest are mask tokens.
        # Positions = arange(block_size) + seq_len[b].
        noise_ids_np = np.full((B, self.block_size),
                               self.mask_token_id,
                               dtype=np.int32)
        noise_pos_np = np.zeros((B, self.block_size), dtype=np.int32)
        pos_range = np.arange(self.block_size, dtype=np.int32)
        for slot in range(num_reqs):
            noise_pos_np[slot] = pos_range + int(per_slot_seq_len[slot])
        noise_input_ids_batched = device_array(self.mesh, noise_ids_np)
        # Overwrite position 0 of each active slot with its next_token_id.
        for slot in range(num_reqs):
            noise_input_ids_batched = noise_input_ids_batched.at[slot, 0].set(
                next_token_ids[slot])
        noise_positions_batched = device_array(self.mesh, noise_pos_np)

        # 5. Pack (B,)-shaped scalars and slot_idx = identity.
        cache_len_arr = device_array(self.mesh, per_slot_cache_len)
        actual_ctx_count_arr = device_array(self.mesh, per_slot_n_copy)
        slot_idx_arr = device_array(
            self.mesh, np.arange(B, dtype=np.int32))
        target_hidden = (ctx_batched, cache_len_arr, actual_ctx_count_arr,
                         slot_idx_arr)

        # DIAGNOSTIC: Log state for first 30 steps (slot 0 only, for brevity).
        import sys
        if not hasattr(self, '_diag_count'):
            self._diag_count = 0
        self._diag_count += 1
        if self._diag_count <= 30:
            print(f"DIAG step={self._diag_count} num_reqs={num_reqs} B={B} "
                  f"common_padded={common_padded} "
                  f"slot0: cache_len={per_slot_cache_len[0]} "
                  f"n_copy={per_slot_n_copy[0]} "
                  f"seq_len={per_slot_seq_len[0]}",
                  file=sys.stderr, flush=True)

        # 6. Build draft attention metadata — query_start_loc now spans B slots.
        if (not hasattr(self, '_draft_query_start_loc')
                or self._draft_query_start_loc.shape[0] != B + 1):
            self._draft_query_start_loc = jnp.arange(
                B + 1, dtype=jnp.int32) * self.block_size
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id].
            get_cpu_tensor().reshape(-1))
        draft_attn_metadata = replace(
            attn_metadata,
            input_positions=noise_positions_batched.reshape(-1),
            query_start_loc=self._draft_query_start_loc,
            block_tables=device_array(self.mesh, block_tables),
        )

        if not hasattr(self, '_dummy_last_indices'):
            self._dummy_last_indices = jnp.zeros(
                self.runner.max_num_reqs, dtype=jnp.int32)
        dummy_last_indices = self._dummy_last_indices[:num_reqs]

        # Save state for propose() to update per-slot counters.
        self._active_num_reqs = num_reqs

        return (
            target_hidden,
            noise_input_ids_batched,
            dummy_last_indices,
            draft_attn_metadata,
        )

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _sample_block_draft_tokens(
        self,
        target_state: nnx.State,
        hidden_states: jax.Array,
    ) -> jax.Array:
        """Greedy-sample draft tokens using the TARGET model's LM head.

        Accepts either 2D ``(T_noise, D)`` (backward-compat precompile path)
        or 3D ``(B, T_noise, D)`` (batched Phase 4 path). Returns ``(B, K)``
        draft ids where ``K = num_speculative_tokens``. The 2D case is
        treated as ``B=1``.
        """
        if hidden_states.ndim == 2:
            hidden_states = hidden_states[jnp.newaxis, :, :]
        draft_hidden = hidden_states[:, 1:1 + self.num_speculative_tokens, :]
        B, K, D = draft_hidden.shape
        draft_flat = draft_hidden.reshape(B * K, D)
        logits = self.target_compute_logits_fn(target_state, draft_flat, None)
        draft_ids = jnp.argmax(logits, axis=-1).reshape(B, K)
        return lax.with_sharding_constraint(
            draft_ids, NamedSharding(self.mesh, PartitionSpec()))

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Generate all draft tokens in one forward pass."""
        # BYPASS TEST: Call draft model directly with minimal JIT
        # (no out_shardings, no donate_argnums) to isolate the tau bug.
        if not hasattr(self, '_bypass_fn'):
            from flax import nnx
            _graphdef = self.model_fn.args[0]  # Extract from partial

            @jax.jit
            def _bypass_forward(state, kv_caches, input_ids, target_hidden, attn_md):
                model = nnx.merge(_graphdef, state)
                return model(kv_caches, input_ids, target_hidden, attn_md)

            self._bypass_fn = _bypass_forward
            import sys
            print("BYPASS: Using direct JIT (no out_shardings, no donate_argnums)",
                  file=sys.stderr, flush=True)

        draft_kv_caches, hidden_states, _ = self._bypass_fn(
            self.state,
            self._draft_kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
        )

        # Update cached references
        self._draft_kv_caches = draft_kv_caches

        # Update cache_len for every active slot: model wrote
        # actual_ctx_count[slot] + T_noise entries at each slot's offset.
        # These are corrected at the start of the next prepare_inputs to
        # match the actual accepted seq_len.
        _, cache_len_arr, actual_ctx_count_arr, _ = target_hidden_states
        old_cache_len = np.asarray(jax.device_get(cache_len_arr))
        actual_ctx_count = np.asarray(jax.device_get(actual_ctx_count_arr))
        T_noise = self.block_size
        for slot in range(self._active_num_reqs):
            self._cache_len_slot[slot] = (int(old_cache_len[slot])
                                          + int(actual_ctx_count[slot])
                                          + T_noise)

        draft_token_ids = self._sample_block_draft_tokens(
            self.target_state, hidden_states)
        # draft_token_ids: (B, K). Manager only reads [:num_reqs].
        if draft_token_ids.shape[0] > self._active_num_reqs:
            draft_token_ids = draft_token_ids[:self._active_num_reqs]

        # DIAGNOSTIC: Log draft output for slot 0
        import sys
        if hasattr(self, '_diag_count') and self._diag_count <= 30:
            _dt = jax.device_get(draft_token_ids[0])[:5]
            print(f"DIAG propose: num_reqs={self._active_num_reqs} "
                  f"cache_len_after_slot0="
                  f"{self._cache_len_slot[0]} "
                  f"draft_ids_slot0={list(_dt)}",
                  file=sys.stderr, flush=True)

        # Pass the FRAMEWORK kv_caches through unchanged
        return kv_caches, draft_token_ids

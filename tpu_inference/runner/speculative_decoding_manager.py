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

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from tpu_inference.logger import init_logger
from tpu_inference.runner import utils as runner_utils
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.utils import device_array

logger = init_logger(__name__)

if TYPE_CHECKING:
    from tpu_inference.layers.common.attention_metadata import \
        AttentionMetadata
    from tpu_inference.runner.tpu_runner import TPUModelRunner


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on JAX/TPU, containing all necessary indices."""
    draft_token_ids: jnp.ndarray
    draft_token_probs: Optional[jnp.ndarray]
    draft_lengths: jnp.ndarray
    draft_lengths_cpu: np.ndarray
    target_logits_indices: jnp.ndarray
    bonus_logits_indices: jnp.ndarray
    final_logits_indices: jnp.ndarray


class SpeculativeDecodingManager:

    def __init__(self, runner: TPUModelRunner):
        self.runner = runner
        # Cached draft tokens.
        self._draft_token_ids: Optional[list[list[int]]] = None
        # Optional per-token draft probabilities aligned with cached draft IDs.
        self._draft_token_probs: Optional[list[list[float]]] = None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.runner.input_batch.req_ids
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def propose_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        aux_hidden_states: Optional[tuple[jnp.ndarray, ...]],
        attn_metadata: AttentionMetadata,
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        scheduler_output: Optional[VllmSchedulerOutput] = None,
        input_ids: Optional[jnp.ndarray] = None,
    ) -> None:
        if self.runner.speculative_config.method == "ngram":
            assert isinstance(self.runner.drafter, NgramProposer)
            self._draft_token_ids = self.runner.drafter.propose(
                sampled_token_ids[:self.runner.input_batch.num_reqs],
                self.runner.input_batch.num_tokens_no_spec,
                self.runner.input_batch.token_ids_cpu)
            self._draft_token_probs = None
        elif self.runner.speculative_config.method == "eagle3":
            self._draft_token_ids = self.propose_eagle3_draft_token_ids(
                sampled_token_ids,
                aux_hidden_states,
                attn_metadata,
                spec_decode_metadata,
                scheduler_output,
                input_ids,
            )
        elif self.runner.speculative_config.method == "dflash":
            # DFlash reuses the same propose flow as Eagle3
            self._draft_token_ids = self.propose_eagle3_draft_token_ids(
                sampled_token_ids,
                aux_hidden_states,
                attn_metadata,
                spec_decode_metadata,
                scheduler_output,
                input_ids,
            )
        else:
            raise NotImplementedError(
                f"Speculative decoding method "
                f"'{self.runner.speculative_config.method}' is not supported.")

    def propose_eagle3_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        aux_hidden_states: Optional[tuple[jnp.ndarray, ...]],
        attn_metadata: AttentionMetadata,
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        scheduler_output: VllmSchedulerOutput,
        input_ids: jnp.ndarray,
    ) -> list[list[int]]:
        # Supports both Eagle3Proposer and DFlashProposer (same interface)
        assert hasattr(self.runner.drafter, 'prepare_inputs') and hasattr(
            self.runner.drafter, 'propose')

        # TODO(woosuk): Refactor the loop.
        req_ids = self.runner.input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = self.runner.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)

        # Pad the batch size to match with existing padding for target model
        pad_len = attn_metadata.seq_lens.shape[0] - len(next_token_ids)
        assert pad_len >= 0
        next_token_ids += [0] * pad_len

        next_token_ids = device_array(
            self.runner.mesh, np.array(next_token_ids, dtype=jnp.int32))

        if spec_decode_metadata is None:
            num_rejected_tokens = None
        else:
            num_draft_tokens = spec_decode_metadata.draft_lengths_cpu
            num_rejected_tokens = [
                int(n) + 1 - len(sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]

            pad_len = self.runner.max_num_reqs - len(num_rejected_tokens)
            num_rejected_tokens += [0] * pad_len
            num_rejected_tokens = device_array(
                self.runner.mesh, np.array(num_rejected_tokens,
                                           dtype=jnp.int32))

        # Use the actual accepted seq_len (num_tokens_no_spec) instead of
        # attn_metadata.seq_lens which includes unverified draft tokens.
        # The attn_metadata.seq_lens = num_computed + num_scheduled, where
        # num_scheduled includes all draft tokens being verified. But the
        # proposer needs the ACCEPTED count to correctly track its context
        # buffer and KV cache positions.
        accepted_seq_lens = self.runner.input_batch.num_tokens_no_spec[
            :attn_metadata.seq_lens.shape[0]].copy()
        accepted_attn_metadata = replace(
            attn_metadata,
            seq_lens=device_array(
                self.runner.mesh,
                accepted_seq_lens.astype(np.int32)),
        )

        target_hidden_states, input_ids, last_token_indices, attn_metadata = self.runner.drafter.prepare_inputs(
            accepted_attn_metadata,
            input_ids,
            aux_hidden_states,
            next_token_ids,
            num_rejected_tokens,
        )

        propose_output = self.runner.drafter.propose(
            kv_caches=self.runner.kv_caches,
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            last_token_indices=last_token_indices,
            target_hidden_states=target_hidden_states,
        )
        if len(propose_output) == 3:
            self.runner.kv_caches, draft_token_ids, draft_token_probs = (
                propose_output)
        else:
            self.runner.kv_caches, draft_token_ids = propose_output
            draft_token_probs = None
        draft_token_ids = np.array(draft_token_ids)
        if draft_token_ids.ndim == 1:
            draft_token_ids = np.expand_dims(draft_token_ids, axis=-1)
        if draft_token_probs is None:
            self._draft_token_probs = None
        else:
            draft_token_probs = np.asarray(draft_token_probs,
                                           dtype=np.float32)
            if draft_token_probs.ndim == 1:
                draft_token_probs = np.expand_dims(draft_token_probs, axis=-1)
            self._draft_token_probs = draft_token_probs.tolist()
        return draft_token_ids.tolist()

    def get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)
        arange = np.concatenate(
            [self.runner.arange_cpu[:n] for n in num_sampled_tokens])
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange
        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # arange: [0, 1, 2, 0, 1, 0]
        arange = np.concatenate(
            [self.runner.arange_cpu[:n] for n in num_draft_tokens])
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.runner.input_ids_cpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        padded_logits_length = runner_utils.get_padded_token_len(
            self.runner.num_logits_paddings, logits_indices.shape[0])
        padded_logits_indices = np.concatenate([
            logits_indices,
            np.zeros(padded_logits_length - logits_indices.shape[0],
                     dtype=np.int32)
        ])

        assert bonus_logits_indices.shape[0] <= padded_num_reqs, (
            f"bonus_logits_indices.shape[0]={bonus_logits_indices.shape[0]} "
            f"padded_num_reqs={padded_num_reqs}")

        padded_bonus_logits_indices = np.concatenate([
            bonus_logits_indices,
            np.zeros(padded_num_reqs - bonus_logits_indices.shape[0],
                     dtype=np.int32)
        ])
        padded_num_draft_tokens = np.concatenate([
            num_draft_tokens,
            np.zeros(padded_num_reqs - num_draft_tokens.shape[0],
                     dtype=np.int32)
        ])
        padded_draft_token_ids = np.concatenate([
            draft_token_ids,
            np.zeros(padded_logits_length - draft_token_ids.shape[0],
                     dtype=np.int32)
        ])
        padded_target_logits_indices = np.concatenate([
            target_logits_indices,
            np.zeros(padded_logits_length - target_logits_indices.shape[0],
                     dtype=np.int32)
        ])

        padded_num_draft_tokens_cpu = padded_num_draft_tokens
        padded_draft_token_probs = None
        if self._draft_token_probs is not None:
            draft_token_probs: list[float] = []
            for req_idx, num_tokens in enumerate(num_draft_tokens):
                num_tokens = int(num_tokens)
                if num_tokens <= 0:
                    continue
                req_probs = []
                if req_idx < len(self._draft_token_probs):
                    req_probs = self._draft_token_probs[req_idx]
                req_probs = [float(x) for x in req_probs[:num_tokens]]
                if len(req_probs) < num_tokens:
                    req_probs.extend([1.0] * (num_tokens - len(req_probs)))
                draft_token_probs.extend(req_probs)
            padded_draft_token_probs = np.concatenate([
                np.asarray(draft_token_probs, dtype=np.float32),
                np.ones(
                    padded_logits_length - len(draft_token_probs),
                    dtype=np.float32,
                ),
            ])

        # CPU -> TPU copy.
        (padded_num_draft_tokens, padded_draft_token_ids,
         padded_logits_indices, padded_target_logits_indices,
         padded_bonus_logits_indices) = device_array(
             self.runner.mesh,
             (padded_num_draft_tokens, padded_draft_token_ids,
              padded_logits_indices, padded_target_logits_indices,
              padded_bonus_logits_indices))
        if padded_draft_token_probs is not None:
            padded_draft_token_probs = device_array(self.runner.mesh,
                                                    padded_draft_token_probs)

        metadata = SpecDecodeMetadata(
            draft_token_ids=padded_draft_token_ids,
            draft_token_probs=padded_draft_token_probs,
            draft_lengths=padded_num_draft_tokens,
            draft_lengths_cpu=padded_num_draft_tokens_cpu,
            target_logits_indices=padded_target_logits_indices,
            bonus_logits_indices=padded_bonus_logits_indices,
            final_logits_indices=padded_logits_indices,
        )
        self._draft_token_probs = None
        return metadata

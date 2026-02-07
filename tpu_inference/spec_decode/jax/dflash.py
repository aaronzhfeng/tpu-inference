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
"""Implements the DFlash proposer for speculative decoding on JAX/TPU."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer


class DFlashProposer(Eagle3Proposer):
    """A DFlash proposer implementation.

    The integration currently reuses the same request batching and proposal loop
    contract as Eagle3 while running a DFlash-specific draft model.
    """

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _get_draft_token_ids(self, _state: nnx.State,
                             hidden_states: jax.Array) -> jax.Array:
        # DFlash uses target LM-head logits for token selection.
        lora_metadata = None
        logits = self.runner.compute_logits_fn(self.runner.state, hidden_states,
                                               lora_metadata)
        draft_token_ids = jnp.argmax(logits, axis=-1)
        return lax.with_sharding_constraint(
            draft_token_ids, NamedSharding(self.mesh, PartitionSpec()))

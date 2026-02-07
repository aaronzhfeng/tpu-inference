# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.spec_decode.jax.dflash import DFlashProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    return jax.sharding.Mesh(devices, axis_names=("model", ))


def test_get_draft_token_ids_uses_target_model_logits():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()

    call_record = {}
    target_state = object()

    def fake_compute_logits_fn(state, hidden_states, lora_metadata):
        call_record["state"] = state
        call_record["shape"] = hidden_states.shape
        call_record["lora_metadata"] = lora_metadata
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]],
                         dtype=jnp.float32)

    proposer.runner = type("Runner", (), {
        "state": target_state,
        "compute_logits_fn": fake_compute_logits_fn,
    })()

    hidden_states = jnp.ones((2, 8), dtype=jnp.bfloat16)
    draft_token_ids = proposer._get_draft_token_ids(None, hidden_states)

    np.testing.assert_array_equal(np.asarray(draft_token_ids),
                                  np.array([1, 0], dtype=np.int32))
    assert call_record["state"] is target_state
    assert call_record["shape"] == hidden_states.shape
    assert call_record["lora_metadata"] is None


def test_get_draft_token_ids_returns_1d_int_ids():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()

    proposer.runner = type("Runner", (), {
        "state": object(),
        "compute_logits_fn": lambda _state, _hidden, _lora: jnp.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
    })()

    hidden_states = jnp.ones((2, 4), dtype=jnp.bfloat16)
    draft_token_ids = proposer._get_draft_token_ids(None, hidden_states)

    assert draft_token_ids.ndim == 1
    assert draft_token_ids.shape == (2, )
    assert jnp.issubdtype(draft_token_ids.dtype, jnp.integer)

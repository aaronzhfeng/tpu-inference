# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.spec_decode.jax.dflash import DFlashProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    return jax.sharding.Mesh(devices, axis_names=("model", ))


def test_sample_block_draft_tokens_greedy():
    """Verify _sample_block_draft_tokens returns argmax of logits."""
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2

    # Fake compute_logits_fn: returns fixed logits for the draft slice
    def fake_compute_logits_fn(state, hidden_states, lora_metadata):
        # hidden_states will be [1:3] of the input (2 tokens)
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]],
                         dtype=jnp.float32)

    proposer.compute_logits_fn = fake_compute_logits_fn

    # hidden_states: block_size=3 tokens, only [1:3] are sampled
    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    state = None  # Not used by fake_compute_logits_fn
    draft_token_ids = proposer._sample_block_draft_tokens(
        state, hidden_states)

    np.testing.assert_array_equal(np.asarray(draft_token_ids),
                                  np.array([1, 0], dtype=np.int32))


def test_sample_block_draft_tokens_shape_and_dtype():
    """Verify output is 1D int array of length num_speculative_tokens."""
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2

    proposer.compute_logits_fn = lambda _state, _hidden, _lora: jnp.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    hidden_states = jnp.ones((3, 4), dtype=jnp.bfloat16)
    draft_token_ids = proposer._sample_block_draft_tokens(
        None, hidden_states)

    assert draft_token_ids.ndim == 1
    assert draft_token_ids.shape == (2, )
    assert jnp.issubdtype(draft_token_ids.dtype, jnp.integer)


def test_build_noise_block():
    """Verify _build_noise_block creates correct noise IDs and positions."""
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()

    seq_len_arr = jnp.array([10], dtype=jnp.int32)
    next_token_ids = jnp.array([42], dtype=jnp.int32)
    mask_token_id = 0
    block_size = 4

    noise_ids, noise_positions = proposer._build_noise_block(
        seq_len_arr, next_token_ids, mask_token_id, block_size)

    # First token should be the next_token_id, rest should be mask
    expected_ids = np.array([42, 0, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(noise_ids), expected_ids)

    # Positions should start at seq_len
    expected_positions = np.array([10, 11, 12, 13], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(noise_positions),
                                  expected_positions)


def test_next_padded_size():
    """Verify _next_padded_size returns correct power-of-2 sizes."""
    assert DFlashProposer._next_padded_size(0) == 16
    assert DFlashProposer._next_padded_size(1) == 16
    assert DFlashProposer._next_padded_size(16) == 16
    assert DFlashProposer._next_padded_size(17) == 32
    assert DFlashProposer._next_padded_size(32) == 32
    assert DFlashProposer._next_padded_size(33) == 64
    assert DFlashProposer._next_padded_size(128) == 128
    assert DFlashProposer._next_padded_size(129) == 256

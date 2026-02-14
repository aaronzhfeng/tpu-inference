# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Llama DFlash draft (target layer ids and config-driven resolution)."""

from types import SimpleNamespace

import pytest

from tpu_inference.models.jax.llama_dflash import _get_dflash_target_layer_ids


def test_get_dflash_target_layer_ids_explicit_config():
    """When dflash_config.target_layer_ids is set, use it."""
    cfg = SimpleNamespace(
        dflash_config={"target_layer_ids": [1, 8, 15, 22, 29]},
        num_target_layers=32,
        num_hidden_layers=5,
    )
    assert _get_dflash_target_layer_ids(cfg, 32) == [1, 8, 15, 22, 29]


def test_get_dflash_target_layer_ids_fallback_single():
    """Fallback with num_target_layers gives middle layer."""
    cfg = SimpleNamespace(
        dflash_config={},
        num_hidden_layers=5,
        num_target_layers=1,
    )
    assert _get_dflash_target_layer_ids(cfg, 32) == [16]


def test_get_dflash_target_layer_ids_fallback_spread():
    """Fallback spreads layers across target depth."""
    cfg = SimpleNamespace(
        dflash_config=None,
        num_hidden_layers=5,
        num_target_layers=4,
    )
    ids = _get_dflash_target_layer_ids(cfg, 32)
    assert len(ids) == 4
    assert ids[0] >= 1 and ids[-1] <= 29
    assert ids == sorted(ids)

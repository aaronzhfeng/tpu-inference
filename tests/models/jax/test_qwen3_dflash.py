# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from tpu_inference.models.jax.qwen3 import (
    _get_dflash_target_layer_ids as get_target_layer_ids_for_qwen3,
)
from tpu_inference.models.jax.qwen3_dflash import (
    _build_target_layer_ids,
    _get_dflash_target_layer_ids as get_target_layer_ids_for_qwen3_dflash,
)


def test_build_target_layer_ids_default_layout():
    assert _build_target_layer_ids(32, 1) == [16]
    assert _build_target_layer_ids(32, 4) == [1, 10, 20, 29]


def test_get_target_layer_ids_prefers_explicit_config():
    cfg = SimpleNamespace(
        dflash_config={"target_layer_ids": [2, 6, 10]},
        num_target_layers=32,
        num_hidden_layers=3,
    )

    assert get_target_layer_ids_for_qwen3_dflash(cfg, 32) == [2, 6, 10]
    assert get_target_layer_ids_for_qwen3(32, cfg) == [2, 6, 10]


def test_get_target_layer_ids_fallback_matches_between_modules():
    cfg = SimpleNamespace(
        dflash_config=None,
        num_target_layers=32,
        num_hidden_layers=3,
    )

    dflash_ids = get_target_layer_ids_for_qwen3_dflash(cfg, 32)
    qwen3_ids = get_target_layer_ids_for_qwen3(32, cfg)

    assert dflash_ids == [1, 15, 29]
    assert qwen3_ids == dflash_ids

"""Torchax wrapper for the PyTorch DFlash draft model.

Loads the original HuggingFace DFlash model (pure PyTorch) and wraps it
so it can run on TPU via torchax + JAX JIT.  This avoids rewriting the
model in JAX and guarantees numerical equivalence with the GPU reference.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import torch
import torch.nn
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from transformers import AutoModel

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    shard_model_to_tpu,
)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Thin runner that exposes forward / combine / logits as a single Module
# ---------------------------------------------------------------------------

class _DFlashRunner(torch.nn.Module):
    """Wrapper that adapts the HF DFlash model for ``functional_call``."""

    def __init__(self, dflash_model: torch.nn.Module):
        super().__init__()
        self.dflash = dflash_model

    def forward(self, **kwargs) -> torch.Tensor:
        if "hidden_state" in kwargs:
            # compute_logits path
            return self._compute_logits(kwargs["hidden_state"],
                                        kwargs["embed_weight"])
        elif "raw_hidden" in kwargs:
            # combine_hidden_states path
            return self._combine_hidden(kwargs["raw_hidden"])
        else:
            # draft forward path
            return self._draft_forward(
                kwargs["noise_embedding"],
                kwargs["target_hidden"],
                kwargs["position_ids"],
            )

    # ----- draft forward --------------------------------------------------

    def _draft_forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the DFlash model (no KV cache, no causal mask).

        Args:
            noise_embedding: (1, block_size, D) embedded noise block.
            target_hidden: (1, ctx_len, num_layers*D) raw concatenated
                           target hidden states (NOT yet projected).
            position_ids: (1, ctx_len + block_size) positions for RoPE
                          covering both context and noise.
        Returns:
            hidden_states: (1, block_size, D) – the draft model output
                           after the final norm (before lm_head).
        """
        return self.dflash(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )

    # ----- combine hidden states ------------------------------------------

    def _combine_hidden(self, raw_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target hidden states through fc + norm."""
        return self.dflash.hidden_norm(self.dflash.fc(raw_hidden))

    # ----- compute logits -------------------------------------------------

    @staticmethod
    def _compute_logits(
        hidden_state: torch.Tensor,
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Logits via tied embeddings: hidden @ embed^T."""
        return torch.nn.functional.linear(hidden_state, embed_weight)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class DFlashTorchaxWrapper:
    """Load the HF DFlash model on CPU, shard to TPU, expose JIT-compiled
    pure functions that the DFlash proposer can call."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.model: _DFlashRunner | None = None
        self.params: dict | None = None  # JAX-land params
        self.embed_weight_jax: jax.Array | None = None  # shared from target

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        draft_model_path: str,
        target_model_state: Any,
    ) -> None:
        """Load HF DFlash model, shard weights to TPU, share embeddings."""

        logger.info("Loading DFlash PyTorch model via AutoModel from %s",
                     draft_model_path)

        with jax.default_device(jax.devices("cpu")[0]):
            hf_model = AutoModel.from_pretrained(
                draft_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # pure-PyTorch ops for torchax
            )
            hf_model.eval()

        self.model = _DFlashRunner(hf_model)
        self.params = shard_model_to_tpu(self.model, self.mesh)
        # shard_model_to_tpu returns torchax tensors; convert to JAX view
        # so JAX JIT can trace through them.
        self.params = jax_view(self.params)

        # Share embedding from the target model (for compute_logits).
        # target_model_state is an nnx.State or similar; extract the
        # embedding weight as a plain JAX array (not nnx.Param).
        embed = getattr(target_model_state.model, "embed_tokens", None)
        if embed is None:
            embed = getattr(target_model_state.model, "embed", None)
        if embed is not None:
            if hasattr(embed, "embedding"):
                w = embed.embedding
                # Unwrap nnx.Param → raw jax.Array
                self.embed_weight_jax = w.value if hasattr(w, "value") else w
            elif hasattr(embed, "weight"):
                w = embed.weight
                if hasattr(w, "value"):
                    self.embed_weight_jax = w.value
                elif isinstance(w, torch.Tensor):
                    self.embed_weight_jax = jax_view(w)
                else:
                    self.embed_weight_jax = w
        if self.embed_weight_jax is None:
            raise RuntimeError(
                "Could not find target model embedding to share with DFlash")

        logger.info("DFlash torchax wrapper loaded successfully.")

    # ------------------------------------------------------------------
    # JIT-compiled callables
    # ------------------------------------------------------------------

    def get_draft_forward_fn(self):
        """Return a JIT-compiled draft forward function.

        Signature::

            draft_forward(params, noise_input_ids, target_hidden,
                          position_ids, embed_weight) -> hidden_states
        """
        model = self.model

        hidden_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA, None))

        @functools.partial(jax.jit, out_shardings=hidden_sharding)
        def draft_forward(
            params: dict,
            noise_input_ids: jax.Array,
            target_hidden: jax.Array,
            position_ids: jax.Array,
            embed_weight: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                # Embed the noise block using the shared embedding
                noise_ids_t = torch_view(noise_input_ids).unsqueeze(0)  # (1, B)
                embed_w_t = torch_view(embed_weight)
                noise_emb = torch.nn.functional.embedding(
                    noise_ids_t, embed_w_t)  # (1, B, D)

                target_h = torch_view(target_hidden).unsqueeze(0)  # (1, C, D')
                pos = torch_view(position_ids).unsqueeze(0)        # (1, C+B)

                output = torch.func.functional_call(
                    model, p,
                    kwargs={
                        "noise_embedding": noise_emb,
                        "target_hidden": target_h,
                        "position_ids": pos,
                    },
                    tie_weights=False,
                )
                # output: (1, block_size, D) → squeeze batch dim
                return jax_view(output.squeeze(0))

        return draft_forward

    def get_combine_hidden_fn(self):
        """Return a JIT-compiled combine_hidden_states function.

        Signature::

            combine_fn(params, raw_hidden) -> projected_hidden
        """
        model = self.model

        hidden_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA, None))

        @functools.partial(jax.jit, out_shardings=hidden_sharding)
        def combine_fn(
            params: dict,
            raw_hidden: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                h = torch_view(raw_hidden)
                out = torch.func.functional_call(
                    model, p,
                    kwargs={"raw_hidden": h},
                    tie_weights=False,
                )
                return jax_view(out)

        return combine_fn

    def get_compute_logits_fn(self):
        """Return a JIT-compiled compute_logits function.

        Signature::

            logits_fn(params, hidden_states, embed_weight) -> logits
        """
        model = self.model

        logits_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA,
                          ShardingAxisName.MLP_TENSOR))

        @functools.partial(jax.jit, out_shardings=logits_sharding)
        def logits_fn(
            params: dict,
            hidden_states: jax.Array,
            embed_weight: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                h = torch_view(hidden_states)
                w = torch_view(embed_weight)
                out = torch.func.functional_call(
                    model, p,
                    kwargs={"hidden_state": h, "embed_weight": w},
                    tie_weights=False,
                )
                return jax_view(out)

        return logits_fn

# SPDX-License-Identifier: Apache-2.0
"""Fused TP all-reduce + residual add + Gemma RMSNorm + per-1x128 FP8 quant.

Single-kernel replacement for the three-kernel decode sequence

    all_reduce(hidden) -> + residual -> Gemma RMSNorm -> per-1x128 FP8 quant

on gfx950 at TP4. The kernel reads peers' staging buffers directly over HIP IPC
and performs its own arrival/completion handshake, so there is no separate
collective launch.

Scope, enforced by :func:`is_supported`:
  * gfx950 (CDNA4) only -- the kernel uses cdna4 ``buffer_load``/``buffer_store``
    and wave64 GCN assembly.
  * TP world size exactly 4 -- the reduction is hand-unrolled over 4 peers and
    the epoch protocol counts in units of 3 remote peers.
  * hidden size exactly 2048, eps 1e-6, and M in {1,2,4,8,16,32,64}.
  * ``_use_aiter_bpreshuffle_gfx95`` must be False. On ROCm >= 7.2 SGLang
    physically preshuffles FP8 weights into the gfx95 bpreshuffle layout, which
    this kernel's consumers do not expect. Returning False here makes the caller
    fall back rather than produce wrong numbers.

Anything outside that envelope returns False and the caller keeps the stock
path. Opt out entirely with ``SGLANG_DISABLE_GLUON_TP_AR_NORM_QUANT=1``.

The kernel body in ``gluon_tp_ar_norm_quant_kernel.py`` is derived from the
Artemis MI355X kernel pack for Qwen3-Next.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.hip_ipc import (
    close_shared_tensor,
    create_shared_tensor,
)
from sglang.srt.utils import get_bool_env_var, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

TP_SIZE = 4
HIDDEN_SIZE = 2048
EPS = 1.0e-6
SUPPORTED_M = (1, 2, 4, 8, 16, 32, 64)
GROUP_SIZE = 128
# Words 0/1 count arrivals / completed reads; word 2 is local CTA progress;
# word 3 is unused. They must start at zero and persist across calls.
SYNC_WORDS = 4
# One lock row per CUDA-graph call site, reserved before capture so the pointer
# table baked into the captured launch stays valid.
CAPTURE_SITE_CAPACITY = 2048


def _gluon_available() -> bool:
    """Probe for Gluon + the CDNA4 intrinsics the kernel needs.

    Mirrors the probe idiom in ``aiter_mla_gluon.py``: SGLang does not pin a
    Triton version that guarantees Gluon (the published ROCm wheel declares
    triton 3.5.1), so this must be a runtime capability check, never an import
    dependency.
    """
    try:
        import triton.experimental.gluon  # noqa: F401
        from triton.experimental.gluon.language.amd.cdna4 import (  # noqa: F401
            buffer_load,
            buffer_store,
        )
    except Exception as exc:  # pragma: no cover - depends on installed triton
        logger.debug("Gluon TP AR+norm+quant unavailable: %s", exc)
        return False
    return True


def _arch_is_gfx950(device: torch.device) -> bool:
    try:
        name = torch.cuda.get_device_properties(device).gcnArchName
    except Exception:  # pragma: no cover
        return False
    return name.split(":", 1)[0] == "gfx950"


def is_supported(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    world_size: int,
    group_size: int = GROUP_SIZE,
) -> bool:
    """Support predicate. False means "caller should use the stock path"."""
    if not _is_hip or residual is None:
        return False
    if get_bool_env_var("SGLANG_DISABLE_GLUON_TP_AR_NORM_QUANT", default="false"):
        return False
    if world_size != TP_SIZE:
        return False
    if hidden_states.dim() != 2 or hidden_states.shape[-1] != HIDDEN_SIZE:
        return False
    if hidden_states.shape[0] not in SUPPORTED_M:
        return False
    if group_size != GROUP_SIZE:
        return False
    if hidden_states.dtype is not torch.bfloat16:
        return False
    if residual.shape != hidden_states.shape or residual.dtype is not torch.bfloat16:
        return False
    if weight.numel() != HIDDEN_SIZE:
        return False
    # The kernel bakes eps into the launch; only the Qwen3-Next value is tuned.
    if abs(eps - EPS) > 1e-12:
        return False
    # On ROCm >= 7.2 SGLang preshuffles FP8 weights; this path is not validated
    # against that layout and would silently produce wrong results.
    from sglang.srt.layers.quantization.fp8_utils import (
        _use_aiter_bpreshuffle_gfx95,
    )

    if _use_aiter_bpreshuffle_gfx95:
        return False
    if not _arch_is_gfx950(hidden_states.device):
        return False
    return _gluon_available()


class GluonTpArNormQuantState:
    """Rendezvous + peer pointer tables for the fused collective.

    Allocates, once per TP group:
      * a staging buffer that peers read this rank's pre-reduction activations from
      * one 4-word synchronization row for eager calls
      * ``CAPTURE_SITE_CAPACITY`` further rows, one per CUDA-graph call site

    and publishes the peer pointer tables the kernel indexes.

    Two table orderings, both load-bearing:
      * ``input_ptrs`` is in **global rank order** -- entry i is rank i's staging
        buffer, and the kernel sums entries 0..3 in that order to preserve the
        production reduction tree.
      * ``lock_ptrs`` is rotated **local-rank-first** -- entry 0 is always this
        rank's own lock row, entries 1..3 the three remote peers. The kernel
        relies on this: it notifies lanes 1..3 (the peers that are not self) and
        polls entry 0 for its own arrival counter.
    """

    def __init__(self, group: ProcessGroup, device: torch.device, max_rows: int):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size != TP_SIZE:
            raise ValueError(f"expected TP{TP_SIZE}, got {self.world_size}")
        self.max_rows = max_rows
        self._closed = False
        lock_bytes = SYNC_WORDS * torch.int32.itemsize

        self.staging, staging_peers, b0 = create_shared_tensor(
            (max_rows, HIDDEN_SIZE), torch.bfloat16, device, group=group
        )
        self._lock, lock_peers, b1 = create_shared_tensor(
            (SYNC_WORDS,), torch.int32, device, group=group
        )
        self._capture_locks, capture_peers, b2 = create_shared_tensor(
            (CAPTURE_SITE_CAPACITY, SYNC_WORDS), torch.int32, device, group=group
        )
        self._opened_bases = b0 + b1 + b2

        self.input_ptr_table = torch.tensor(
            staging_peers, dtype=torch.uint64, device=device
        )
        self.lock_ptr_table = torch.tensor(
            self._rotate(lock_peers), dtype=torch.uint64, device=device
        )
        # Row `site` of every peer's capture-lock allocation, rotated the same way.
        self.capture_lock_tables = torch.tensor(
            [
                self._rotate([base + site * lock_bytes for base in capture_peers])
                for site in range(CAPTURE_SITE_CAPACITY)
            ],
            dtype=torch.uint64,
            device=device,
        )
        self._next_site = 0
        torch.cuda.synchronize(device)

    def _rotate(self, ptrs: List[int]) -> Tuple[int, ...]:
        """Reorder global-rank-ordered pointers to local-rank-first."""
        return tuple(ptrs[(self.rank + offset) % self.world_size] for offset in range(self.world_size))

    def reserve_site(self) -> int:
        """Claim a lock row for one CUDA-graph call site."""
        if self._next_site >= CAPTURE_SITE_CAPACITY:
            raise RuntimeError("Gluon TP AR+norm+quant graph site capacity exceeded")
        site = self._next_site
        self._next_site += 1
        return site

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_shared_tensor(self._opened_bases)


def fused_tp_ar_add_gemma_rmsnorm_group_fp8_quant(
    state: GluonTpArNormQuantState,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    site: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused collective.

    Returns ``(fp8_out, scale_out, residual_out, bf16_out)``. ``bf16_out`` is the
    pre-quantization normed activation, which GDN-style layers need for their
    bf16 gating projection (``keep_bf16=True`` in the caller); it is produced by
    the same kernel, so obtaining it costs nothing extra.
    """
    from sglang.srt.distributed.device_communicators.gluon_tp_ar_norm_quant_kernel import (
        tp4_allreduce_add_gemma_rmsnorm_group_fp8_quant_gluon,
    )

    rows = hidden_states.shape[0]
    state.staging[:rows].copy_(hidden_states)
    lock_table = (
        state.lock_ptr_table if site is None else state.capture_lock_tables[site]
    )
    normalized, residual_out, quantized, scales, _reduced = (
        tp4_allreduce_add_gemma_rmsnorm_group_fp8_quant_gluon(
            state.staging[:rows],
            state.input_ptr_table,
            lock_table,
            residual,
            weight,
            eps=EPS,
        )
    )
    return quantized, scales, residual_out, normalized

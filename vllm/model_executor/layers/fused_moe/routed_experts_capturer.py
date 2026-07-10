# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# Constants
_TMP_DIR = tempfile.gettempdir()
_LOCK_FILE_PREFIX = os.path.join(_TMP_DIR, "vllm_routed_experts")
_BUFFER_PREFIX = "vllm_routed_experts_buffer"

# Global singleton instances
_global_experts_capturer: RoutedExpertsCapturer | None = None
_global_experts_reader: RoutedExpertsReader | None = None


def _get_compress_ratio(kv_cache_spec) -> int:
    """Extract compress_ratio from a kv_cache_spec.

    The scheduler receives individual specs (e.g. AttentionSpec) after
    ``generate_scheduler_kv_cache_config`` flattens
    ``UniformTypeKVCacheSpecs``.  Workers, however, still hold the
    ``UniformTypeKVCacheSpecs`` wrapper which does **not** expose
    ``compress_ratio`` directly.  In that case we look it up from the
    first inner spec.
    """
    cr = getattr(kv_cache_spec, "compress_ratio", None)
    if cr is not None:
        return cr
    # kv_cache_spec may be UniformTypeKVCacheSpecs (worker side)
    inner = getattr(kv_cache_spec, "kv_cache_specs", None)
    if inner:
        first_spec = next(iter(inner.values()), None)
        if first_spec is not None:
            return getattr(first_spec, "compress_ratio", 1)
    return 1


@contextmanager
def _file_lock(lock_file: str, mode: str = "wb+") -> Generator[None, None, None]:
    """Context manager for file-based locking."""
    with open(lock_file, mode) as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)


def _create_or_attach_shared_memory(
    name: str, size: int, lock_file: str
) -> tuple[shared_memory.SharedMemory, bool]:
    """Create or attach to shared memory with proper locking.

    Returns:
        Tuple of (SharedMemory instance, whether it was newly created).
    """
    # Ensure lock file exists before acquiring lock
    with open(lock_file, "wb"):
        pass

    with _file_lock(lock_file):
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            return shm, True
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name, create=False, size=size)

        if shm.size != size:
            logger.warning(
                "Shared memory %s size mismatch; recreating",
                name,
            )
            shm.close()
            shm.unlink()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                logger.info("Created shared memory %s", name)
                return shm, True
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False, size=size)
                logger.info("Linked to existing shared memory %s", name)

    return shm, False


class RoutedExpertsCapturer:
    """
    Capturer for routed experts with device and optional shared memory buffer.

    This class captures expert routing decisions during model forward passes
    and optionally stores them in shared memory for cross-process access.
    """

    _instance: RoutedExpertsCapturer | None = None

    def __init__(self) -> None:
        self._device_buffer: torch.Tensor | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None
        self._lock_file: str | None = None

    @classmethod
    def create(cls) -> RoutedExpertsCapturer:
        """Create a global singleton instance."""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")

        _global_experts_capturer = cls()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> RoutedExpertsCapturer | None:
        """Get the global singleton instance."""
        return _global_experts_capturer

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        vllm_config: VllmConfig,
        compress_ratio: int = 1,
    ) -> None:
        """
        Initialize the device buffer and optionally shared memory buffer.

        Args:
            max_num_batched_tokens: Maximum number of tokens in a batch.
            max_num_kv_tokens: Maximum number of KV tokens for shared memory.
            vllm_config: vllm configuration containing layer and expert info.
            compress_ratio: Compression ratio for KV cache (default: 1).
        """

        if self._device_buffer is not None:
            raise RuntimeError("Device buffer has already been initialized")

        hf_config = vllm_config.model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok
        self.compress_ratio = compress_ratio

        # Initialize device buffer
        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        # In expert-parallel setups all EP ranks (each is also a TP rank)
        # share the same dp_rank == 0 shared memory so that every rank
        # can save its own token data.  When EP is disabled, only TP0
        # creates the shared memory.
        enable_ep = vllm_config.parallel_config.enable_expert_parallel
        if not enable_ep and get_tensor_model_parallel_rank() != 0:
            return

        # Initialize shared memory
        max_num_host_slots = max_num_kv_tokens * compress_ratio
        shape = (max_num_host_slots, num_layers, num_experts_per_tok)
        buffer_size = int(np.prod(shape)) * np.dtype(np.int32).itemsize
        instance_id = vllm_config.instance_id
        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{self.dp_rank}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"

        self._shm, newly_created = _create_or_attach_shared_memory(
            shm_name, buffer_size, self._lock_file
        )
        self._host_buffer_view = np.ndarray(shape, dtype=np.int32, buffer=self._shm.buf)
        if newly_created:
            self._host_buffer_view.fill(0)

        logger.debug(
            "Created shared memory buffer '%s' with shape %s compress_ratio=%s",
            shm_name,
            shape,
            compress_ratio,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated before routing.
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside quant_method.apply;
                # select_experts only sees this rank's tokens.
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch dim "
                    f"{n} (expected {total} or {token_num_per_dp} "
                    f"for dp_rank={self.dp_rank})"
                )

        if layer_id >= self._device_buffer.shape[1]:
            return

        self._device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        """Clear the device buffer."""
        if self._device_buffer is not None:
            self._device_buffer.zero_()

    def save_captured_experts(self, indices: np.ndarray, token_positions: np.ndarray | None = None) -> None:
        """
        Save captured experts from device buffer to shared memory.

        Args:
            indices: Array of indices indicating where to store the data.
            token_positions: Original token positions within each request.
                Required when compress_ratio > 1 to expand indices to
                per-token granularity.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            raise RuntimeError("Device buffer not initialized.")

        num_tokens = len(indices)
        data = self._device_buffer[:num_tokens, :, :].cpu().numpy()

        # Expand kv_slot indices to per-token host indices.
        # host_index = kv_slot * compress_ratio + (token_position % compress_ratio)
        # if self.compress_ratio > 1 and token_positions is not None:
        #     host_indices = (indices * self.compress_ratio
        #                    + (token_positions % self.compress_ratio))
        # else:
        #     host_indices = indices
        host_indices = indices

        # Skip slots with -1 (padding tokens that have no KV cache slot).
        valid_mask = host_indices >= 0
        valid_indices = host_indices[valid_mask]
        valid_data = data[valid_mask]

        logger.debug(
            "[routed_experts] save: num_tokens=%d valid=%d "
            "indices[:3]=%s host_indices[:3]=%s data_nonzero=%s "
            "compress_ratio=%s host_buf_shape=%s",
            num_tokens, len(valid_indices),
            indices[:3], valid_indices[:3], (valid_data != 0).any(),
            getattr(self, 'compress_ratio', 'N/A'),
            self._host_buffer_view.shape if self._host_buffer_view is not None else None,
        )

        with _file_lock(self._lock_file):
            self._host_buffer_view[valid_indices, :, :] = valid_data

    def cleanup(self) -> None:
        """Explicitly clean up shared memory resources."""
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                logger.debug("Exception during cleanup for capturer", exc_info=True)
            finally:
                self._shm = None

    def __del__(self) -> None:
        """Clean up shared memory on destruction."""
        self.cleanup()


class RoutedExpertsReader:
    """
    Reader for routed experts from shared memory.

    This class attaches to shared memory created by RoutedExpertsCapturer
    and reads expert routing decisions.

    In expert-parallel (EP) setups, each EP rank writes to its own shared
    memory segment. The reader attaches to *all* EP ranks and merges
    results so that every token's routing data is available regardless of
    which EP rank processed it.
    """

    _instance: RoutedExpertsReader | None = None

    def __init__(self) -> None:
        # Per EP-rank lists (index = dp_rank / ep_rank)
        self._shms: list[shared_memory.SharedMemory] = []
        self._host_buffer_views: list[np.ndarray] = []
        self._lock_files: list[str] = []
        self._dp_size: int = 1

    @classmethod
    def create(cls) -> RoutedExpertsReader:
        """Create a global singleton instance."""
        global _global_experts_reader
        if _global_experts_reader is not None:
            raise RuntimeError("Experts reader already created.")

        _global_experts_reader = cls()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> RoutedExpertsReader | None:
        """Get the global singleton instance."""
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
        return _global_experts_reader

    def attach_buffer(
        self,
        max_num_kv_tokens: int,
        vllm_config: VllmConfig,
        compress_ratio: int = 1,
    ) -> None:
        """
        Attach to shared memory buffer(s).

        In EP setups, attaches to all EP ranks' buffers so that routing
        data is available for every token.

        Args:
            max_num_kv_tokens: Maximum number of KV tokens.
            vllm_config: vllm configuration.
            compress_ratio: Compression ratio for KV cache (default: 1).
        """
        if self._shms:
            logger.warning("Already attached to shared memory buffer.")
            return  # Already attached

        self.compress_ratio = compress_ratio
        hf_config = vllm_config.model_config.hf_text_config
        max_num_host_slots = max_num_kv_tokens * self.compress_ratio
        shape = (
            max_num_host_slots,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
        )

        dp_size = vllm_config.parallel_config.data_parallel_size
        self._dp_size = dp_size
        instance_id = vllm_config.instance_id

        for dp_rank in range(dp_size):
            lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{dp_rank}.lock"
            shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{dp_rank}"

            # The scheduler may start before all EP workers have created
            # their shared memory.  Retry with a short delay.
            shm = None
            for attempt in range(60):  # up to ~30 s
                try:
                    with _file_lock(lock_file, mode="rb+"):
                        with patch(
                            "multiprocessing.resource_tracker.register",
                            lambda *args, **kwargs: None,
                        ):
                            shm = shared_memory.SharedMemory(name=shm_name)
                    break
                except FileNotFoundError:
                    import time
                    time.sleep(0.5)
            if shm is None:
                logger.warning(
                    "RoutedExpertsReader: could not attach to EP rank %d "
                    "shared memory '%s' after 30 s; skipping.",
                    dp_rank, shm_name,
                )
                continue

            buf_view = np.ndarray(
                shape, dtype=np.int32, buffer=shm.buf
            )

            self._shms.append(shm)
            self._host_buffer_views.append(buf_view)
            self._lock_files.append(lock_file)

        logger.info(
            "RoutedExpertsReader attached to %d EP rank buffer(s), "
            "shape=%s compress_ratio=%s",
            dp_size, shape, compress_ratio,
        )

    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray:
        """
        Read routed expert data from shared memory.

        In EP setups, reads from all EP ranks' buffers and merges them
        so that each token gets the routing data from whichever rank
        wrote to its slot.

        Args:
            indices: Array of indices to read.

        Returns:
            Copy of the expert routing data for the given indices.
        """
        if not self._host_buffer_views:
            raise RuntimeError("Buffer not attached. Call attach_buffer() first.")

        # Read from all EP rank buffers and merge.
        # Each EP rank writes to its own slot positions; zeros indicate
        # "no data at this slot from this rank".  We merge by picking
        # the first non-zero entry across ranks.
        result: np.ndarray | None = None
        for dp_rank, (buf_view, lock_file) in enumerate(
            zip(self._host_buffer_views, self._lock_files)
        ):
            with _file_lock(lock_file, mode="rb+"):
                chunk = buf_view[indices, :, :].copy()
            if dp_rank == 0:
                result = chunk
            else:
                # Merge: where result is zero, take chunk's value.
                zero_mask = result == 0
                result[zero_mask] = chunk[zero_mask]

        assert result is not None

        logger.debug(
            "[routed_experts] read: indices[:3]=%s result_nonzero=%s "
            "host_buf_shape=%s dp_size=%s",
            indices[:3], (result != 0).any(),
            self._host_buffer_views[0].shape if self._host_buffer_views else None,
            self._dp_size,
        )

        return result

    def cleanup(self) -> None:
        """Explicitly clean up resources (close without unlink)."""
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                logger.debug("Exception during cleanup for reader", exc_info=True)
        self._shms.clear()
        self._host_buffer_views.clear()
        self._lock_files.clear()

    def __del__(self) -> None:
        """Close shared memory on destruction (do not unlink)."""
        self.cleanup()

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
        max_num_reqs: int = 0,
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

        if get_tensor_model_parallel_rank() != 0:
            return

        # Initialize shared memory
        max_num_host_slots = max_num_kv_tokens * compress_ratio
        shape = (max_num_host_slots, num_layers, num_experts_per_tok)
        # Shared memory layout:
        # [0, 2): metadata (block_table.block_size, block_table.physical_block_size)
        # [2, 2 + data_elems): routed_experts data
        # [2 + data_elems, 2 + data_elems + sm_elems): slot_mapping
        # [2 + data_elems + sm_elems, ...): token_counts_per_req
        elem_size = np.dtype(np.int32).itemsize
        meta_elems = 2
        data_elems = max_num_host_slots * num_layers * num_experts_per_tok
        sm_elems = max_num_batched_tokens
        tc_elems = max_num_reqs
        total_elems = meta_elems + data_elems + sm_elems + tc_elems
        buffer_size = total_elems * elem_size
        instance_id = vllm_config.instance_id
        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{self.dp_rank}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"

        self._shm, newly_created = _create_or_attach_shared_memory(
            shm_name, buffer_size, self._lock_file
        )
        buf = np.ndarray((total_elems,), dtype=np.int32, buffer=self._shm.buf)
        # Metadata: [0] = block_table.block_size, [1] = physical_block_size
        self._metadata_view = buf[:meta_elems]
        # Data view
        self._host_buffer_view = buf[meta_elems:meta_elems + data_elems].reshape(shape)
        # Slot mapping view
        sm_start = meta_elems + data_elems
        self._slot_mapping_view = buf[sm_start:sm_start + sm_elems]
        # Token counts view
        tc_start = sm_start + sm_elems
        self._token_counts_view = buf[tc_start:tc_start + tc_elems]
        if newly_created:
            buf.fill(0)

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
        self._metadata_views: list[np.ndarray] = []
        self._slot_mapping_views: list[np.ndarray] = []
        self._token_counts_views: list[np.ndarray] = []
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

        # Compute shared memory layout (same as worker side)
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok
        scheduler_config = vllm_config.scheduler_config
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_num_reqs = scheduler_config.max_num_seqs

        data_elems = max_num_host_slots * num_layers * num_experts_per_tok
        meta_elems = 2
        sm_elems = max_num_batched_tokens
        tc_elems = max_num_reqs
        total_elems = meta_elems + data_elems + sm_elems + tc_elems

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

            buf = np.ndarray((total_elems,), dtype=np.int32, buffer=shm.buf)
            # Metadata
            meta_view = buf[:meta_elems]
            # Data view
            buf_view = buf[meta_elems:meta_elems + data_elems].reshape(shape)
            # Slot mapping view
            sm_start = meta_elems + data_elems
            sm_view = buf[sm_start:sm_start + sm_elems]
            # Token counts view
            tc_start = sm_start + sm_elems
            tc_view = buf[tc_start:tc_start + tc_elems]

            self._shms.append(shm)
            self._host_buffer_views.append(buf_view)
            self._metadata_views.append(meta_view)
            self._slot_mapping_views.append(sm_view)
            self._token_counts_views.append(tc_view)
            self._lock_files.append(lock_file)

        logger.info(
            "RoutedExpertsReader attached to %d EP rank buffer(s), "
            "shape=%s compress_ratio=%s",
            dp_size, shape, compress_ratio,
        )

    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray:
        """
        Read routed expert data from shared memory.

        Args:
            indices: Array of KV-slot indices to read.

        Returns:
            Copy of the expert routing data for the given indices.
        """
        if not self._host_buffer_views:
            raise RuntimeError("Buffer not attached. Call attach_buffer() first.")

        buf_view = self._host_buffer_views[0]
        lock_file = self._lock_files[0]
        with _file_lock(lock_file, mode="rb+"):
            result = buf_view[indices, :, :].copy()

        logger.debug(
            "[routed_experts] read: indices[:3]=%s result_nonzero=%s",
            indices[:3], (result != 0).any(),
        )

        return result

    def get_routed_experts_by_request(
        self, token_offset: int, token_count: int
    ) -> np.ndarray:
        """
        Read routed expert data for a request using token-order indexing.

        The worker writes the slot_mapping for all tokens in each step
        into shared memory.  The scheduler uses token_offset and
        token_count (derived from cumulative token counts) to find the
        slot indices for a given request, then reads the data.

        Args:
            token_offset: Start index in the token-order slot_mapping.
            token_count: Number of tokens for this request.

        Returns:
            Copy of the expert routing data for the given request.
        """
        if not self._host_buffer_views:
            raise RuntimeError("Buffer not attached. Call attach_buffer() first.")
        if not self._slot_mapping_views:
            raise RuntimeError("Slot mapping not available in shared memory.")

        sm_view = self._slot_mapping_views[0]
        buf_view = self._host_buffer_views[0]
        lock_file = self._lock_files[0]

        with _file_lock(lock_file, mode="rb+"):
            # Read slot indices for this request's tokens
            slot_indices = sm_view[token_offset:token_offset + token_count].copy()

        # Filter out -1 slots (tokens without KV cache slots)
        valid_mask = slot_indices >= 0
        valid_slots = slot_indices[valid_mask]

        result = np.zeros(
            (token_count, buf_view.shape[1], buf_view.shape[2]),
            dtype=np.int32,
        )

        if len(valid_slots) > 0:
            with _file_lock(lock_file, mode="rb+"):
                valid_data = buf_view[valid_slots, :, :].copy()
            result[valid_mask] = valid_data

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

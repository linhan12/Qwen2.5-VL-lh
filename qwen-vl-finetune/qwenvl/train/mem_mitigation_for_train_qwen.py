"""
mem_mitigation_for_train_qwen.py

Helper utilities to mitigate CUDA fragmentation / reserved-memory issues during long-running
transformers / deepspeed training. 

How to use (minimal):
1. Put this file somewhere importable (same folder as train_qwen.py is fine).
2. At the top of your `train_qwen.py`, call:

    from mem_mitigation_for_train_qwen import enable_mem_mitigation, MemoryCleanupCallback
    enable_mem_mitigation()

   And when you construct your `Trainer(...)`, add the callback:

    callbacks = list(existing_callbacks) if existing_callbacks is not None else []
    callbacks.append(MemoryCleanupCallback(check_every_steps=1, verbose_rank_only=True))
    trainer = Trainer(..., callbacks=callbacks)

If you prefer not to modify trainer creation, you can also call `periodic_memory_cleanup()` manually inside your training loop.

What this file does:
- sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation
- provides a TrainerCallback that performs `torch.cuda.ipc_collect()`, `torch.cuda.empty_cache()` and `gc.collect()` on step/epoch boundaries
- prints per-rank memory usage summary so you can observe fragmentation trends

Notes:
- This is a low-risk mitigation; it does not change model code or ZeRO stage. For larger wins, consider batch-size / seq-length / gradient-checkpointing / ZeRO-3.
- Calls to `empty_cache()` and `ipc_collect()` are cheap compared to OOM restarts but may impact perf slightly when called very often. Default is `check_every_steps=1` (every step); you can increase to e.g. 10.

"""

import os
import time
import math
import gc
import sys
import logging
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import CallbackHandler

logger = logging.getLogger(__name__)


def enable_mem_mitigation(expandable_segments: bool = True):
    """Enable environment settings to reduce fragmentation.

    This sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (recommended by PyTorch) if not already set.
    Call this before any CUDA allocation (i.e., at top of script, before model creation).
    """
    env_key = "PYTORCH_CUDA_ALLOC_CONF"
    cur = os.environ.get(env_key, "")
    wanted = "expandable_segments:True"
    if wanted in cur:
        logger.info(f"{env_key} already contains '{wanted}'")
    else:
        new = (cur + "," + wanted) if cur else wanted
        os.environ[env_key] = new
        logger.info(f"Set {env_key}={os.environ[env_key]}")

    # optional: set allocator config so it prints less noisy errors and uses the new allocator
    # Note: must be set before CUDA initialization / model creation


def _mem_summary(prefix: str = "") -> str:
    """Return a short memory summary string for logging."""
    if not torch.cuda.is_available():
        return prefix + " (no CUDA)"
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
    allocated = torch.cuda.memory_allocated(dev) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(dev) / 1024 ** 3
    # fragmentation approximation
    free_approx = reserved - allocated
    return (
        f"{prefix} GPU{dev}: total={total:.2f}GB, allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, reserved-allocated={free_approx:.2f}GB"
    )


class MemoryCleanupCallback(TrainerCallback):
    """A TrainerCallback that attempts to reduce fragmentation by running cleanup ops.

    - Calls torch.cuda.ipc_collect() then torch.cuda.empty_cache() and gc.collect()
    - Logs memory usage periodically

    Parameters
    ----------
    check_every_steps: int
        Run cleanup every N steps (default 1). Use larger values to reduce perf impact.
    verbose_rank_only: bool
        If True, only rank 0 prints verbose logs (useful in multi-rank runs). Set False to have each rank log.
    max_reserved_threshold_gb: Optional[float]
        If set, cleanup will only be triggered when (reserved - allocated) >= threshold.
    """

    def __init__(self, check_every_steps: int = 1, verbose_rank_only: bool = True, max_reserved_threshold_gb: Optional[float] = None):
        self.check_every_steps = max(1, int(check_every_steps))
        self.verbose_rank_only = verbose_rank_only
        self.step_cnt = 0
        self.max_reserved_threshold_gb = max_reserved_threshold_gb

    def _should_log_here(self):
        # try to detect distributed rank for simple gating
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if self.verbose_rank_only:
            return rank == 0
        return True

    def _maybe_cleanup(self):
        if not torch.cuda.is_available():
            return
        # run lightweight collection to help defragmentation
        try:
            # collect any stale IPC handles
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            # free cached blocks back to OS if possible
            torch.cuda.empty_cache()
        except Exception:
            pass
        # python-level garbage
        gc.collect()

    def _maybe_log_mem(self, prefix: str = ""):
        if self._should_log_here():
            logger.info(_mem_summary(prefix=prefix))

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # called at the end of each step
        self.step_cnt += 1
        if (self.step_cnt % self.check_every_steps) != 0:
            return control

        if torch.cuda.is_available():
            # only cleanup when threshold is hit (if configured)
            if self.max_reserved_threshold_gb is not None:
                dev = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(dev) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(dev) / 1024 ** 3
                if (reserved - allocated) < self.max_reserved_threshold_gb:
                    # skip cleanup to avoid perf overhead
                    return control

        # log before/after so you can inspect effect
        self._maybe_log_mem(prefix=f"[step {state.global_step}] before cleanup ->")
        self._maybe_cleanup()
        self._maybe_log_mem(prefix=f"[step {state.global_step}] after  cleanup ->")
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # extra cleanup at epoch end
        self._maybe_log_mem(prefix=f"[epoch {state.epoch}] before epoch-end cleanup ->")
        self._maybe_cleanup()
        self._maybe_log_mem(prefix=f"[epoch {state.epoch}] after  epoch-end cleanup ->")
        return control


def periodic_memory_cleanup(interval_seconds: int = 300, verbose: bool = True):
    """Call this function periodically from your own loop if you don't use TrainerCallback.

    Example:
        last = time.time()
        while training:
            ...
            if time.time() - last > 300:
                periodic_memory_cleanup()
                last = time.time()
    """
    if verbose and _should_log_here_global():
        logger.info("Periodic memory cleanup requested")
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _should_log_here_global():
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank == 0


# If this file is executed directly, print a short diagnostic and enable mitigation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    enable_mem_mitigation()
    logger.info("Memory mitigation enabled (script run directly). Current memory:")
    logger.info(_mem_summary(prefix="[diagnostic] "))

# eof

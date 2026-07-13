"""Loads a frozen OPD teacher's LM head for full-vocab KL reconstruction.

`--teacher-score-mode full_vocab` ships only the teacher's last-layer hidden state per
response position (see `compute_teacher_log_probs` in
orbit/rollout/generate_utils/generate_endpoint_utils.py), not its full vocab logits --
transmitting a vocab-sized vector per token over HTTP would be far more expensive. The
training side reconstructs the teacher's full logits itself by multiplying that hidden
state through the teacher's own LM head (`hidden_state @ lm_head.weight.T`), which this
module loads once per process and caches.
"""

import json
import logging
import os

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

_TEACHER_LM_HEAD_CACHE: dict[str, torch.Tensor] = {}


def _find_weight_key(checkpoint_path: str) -> str:
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return "model.embed_tokens.weight" if config.get("tie_word_embeddings", False) else "lm_head.weight"


def _load_weight_from_safetensors(checkpoint_path: str, weight_key: str) -> torch.Tensor:
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_file = index["weight_map"][weight_key]
    else:
        shard_file = "model.safetensors"

    shard_path = os.path.join(checkpoint_path, shard_file)
    with safe_open(shard_path, framework="pt") as f:
        return f.get_tensor(weight_key)


def load_teacher_lm_head(checkpoint_path: str) -> torch.Tensor:
    """Load (or return the cached) `[vocab_size, hidden_size]` teacher LM head weight.

    Loaded once per process, onto CPU, regardless of its current device -- callers that
    need it on GPU should go through `onload_teacher_lm_head` (see
    orbit/backends/megatron_utils/actor.py's sleep()/wake_up() hooks).
    """
    if checkpoint_path not in _TEACHER_LM_HEAD_CACHE:
        weight_key = _find_weight_key(checkpoint_path)
        weight = _load_weight_from_safetensors(checkpoint_path, weight_key)
        logger.info(
            "Loaded teacher LM head %s (%s) from %s for full-vocab OPD",
            weight_key,
            tuple(weight.shape),
            checkpoint_path,
        )
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = weight
    return _TEACHER_LM_HEAD_CACHE[checkpoint_path]


def offload_teacher_lm_head(checkpoint_path: str) -> None:
    """Move the cached teacher LM head to CPU in place, if it's been loaded."""
    if checkpoint_path in _TEACHER_LM_HEAD_CACHE:
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = _TEACHER_LM_HEAD_CACHE[checkpoint_path].to("cpu")


def onload_teacher_lm_head(checkpoint_path: str, device: torch.device | str) -> None:
    """Move the cached teacher LM head to `device` in place, if it's been loaded."""
    if checkpoint_path in _TEACHER_LM_HEAD_CACHE:
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = _TEACHER_LM_HEAD_CACHE[checkpoint_path].to(device)

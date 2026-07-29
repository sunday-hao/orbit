"""Loads a frozen OPD teacher's LM head for full-vocab KL reconstruction.

`--teacher-score-mode full_vocab` ships only the teacher's last-layer hidden state per
response position (see `compute_teacher_log_probs` in
orbit/rollout/generate_utils/generate_endpoint_utils.py), not its full vocab logits --
transmitting a vocab-sized vector per token over HTTP would be far more expensive. The
training side reconstructs the teacher's full logits itself by multiplying that hidden
state through the teacher's own LM head (`hidden_state @ lm_head.weight.T`), which this
module loads once per process and caches -- sharded along the vocabulary to match Megatron's
partition of the student's output layer when tensor parallelism is on.
"""

import json
import logging
import os
from argparse import Namespace

import torch
from safetensors import safe_open

from .parallel import get_parallel_state
from .vocab_parallel import vocab_shard_bounds

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


def load_teacher_lm_head(args: Namespace) -> torch.Tensor:
    """Load (or return the cached) teacher LM head weight for this rank.

    `[vocab_size, hidden_size]` under tensor_model_parallel_size == 1. Above that, the rows
    are sliced down to the vocabulary columns this rank's output layer owns, so the student
    and teacher sides of the divergence line up column-for-column; ranks whose shard runs
    past the teacher's real vocabulary get correspondingly fewer rows, or none at all.

    Loaded once per process, onto CPU, regardless of its current device -- callers that
    need it on GPU should go through `onload_teacher_lm_head` (see
    orbit/backends/megatron_utils/actor.py's sleep()/wake_up() hooks).
    """
    checkpoint_path = args.teacher_hf_checkpoint
    if checkpoint_path not in _TEACHER_LM_HEAD_CACHE:
        weight_key = _find_weight_key(checkpoint_path)
        weight = _load_weight_from_safetensors(checkpoint_path, weight_key)
        vocab_size = weight.size(0)

        if get_parallel_state().tp.size > 1:
            start, end = vocab_shard_bounds(args.padded_vocab_size)
            # .clone() rather than keeping the slice as a view, so the rows this rank does
            # not own are actually freed -- the point of sharding is to not carry ~1.2 GiB
            # (152k x 2048, fp32) per rank, GPU-resident between wake_up() and sleep().
            weight = weight[start : min(end, vocab_size)].clone()
            logger.info(
                "Loaded teacher LM head %s from %s for full-vocab OPD: vocab [%d, %d) of %d, "
                "shard %s",
                weight_key,
                checkpoint_path,
                start,
                min(end, vocab_size),
                vocab_size,
                tuple(weight.shape),
            )
        else:
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

"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

import asyncio
import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from orbit.utils.http_utils import post
from orbit.utils.processing_utils import encode_image_for_rollout_engine
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

# Element type of the teacher hidden states sglang sends back
_HIDDEN_STATE_DTYPE = np.dtype(np.float32)

_warned_legacy_hidden_states = False


def _warn_legacy_hidden_states_format() -> None:
    """Warn once per process that the teacher is on the slow nested-JSON path.

    Emitted per sample would be one line per request, so this fires a single time and then
    stays quiet.
    """
    global _warned_legacy_hidden_states
    if _warned_legacy_hidden_states:
        return
    _warned_legacy_hidden_states = True
    logger.warning(
        "Teacher returned hidden_states as nested JSON floats rather than a base64 buffer. "
        "This works but is the dominant cost of a full-vocab OPD step: the sglang server "
        "spends minutes per step materializing hundreds of millions of Python floats and "
        "serializing them to multi-GB JSON while its GPU idles."
    )


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]

        # Follow-up shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    return_logprob: bool = True,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": return_logprob,
        "return_routed_experts": args.use_rollout_routing_replay,
    }
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    return payload, None


def should_request_rollout_logprobs(args, evaluation: bool = False) -> bool:
    if getattr(args, "use_orbit_router", False) and "RadixTreeMiddleware" in getattr(
        args, "orbit_router_middleware_paths", []
    ):
        return True
    if evaluation:
        return bool(getattr(args, "eval_return_rollout_logprobs", False))
    return True


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    if args.use_orbit_router and "RadixTreeMiddleware" in args.orbit_router_middleware_paths:
        from orbit.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        # Follow-up may rename to match
        await postprocess_sample_with_radix_tree(args, sample, output)

        assert not update_loss_mask, "This code branch has not implemented update_loss_mask"
    else:
        if "output_token_logprobs" in output["meta_info"]:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            new_response_log_probs = [item[0] for item in output_token_logprobs]
        else:
            output_token_logprobs = None
            new_response_log_probs = None

        if output.get("output_ids") is not None:
            new_response_tokens = output["output_ids"]
        elif output_token_logprobs is not None:
            new_response_tokens = [item[1] for item in output_token_logprobs]
        else:
            new_response_tokens = []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        if new_response_log_probs is not None:
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += new_response_log_probs

        if update_loss_mask:
            if sample.loss_mask is None:
                sample.loss_mask = []
            sample.loss_mask += [1] * len(new_response_tokens)

    # Follow-up handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_rollout_topk_from_response(args, output, sample, "routed_experts")

    # Follow-up may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def get_rollout_topk_from_response(args, output, sample, key):
    info = output["meta_info"].get(key)
    if info is None:
        return None
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    return x.reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)


async def compute_teacher_log_probs(args, model_name: str, samples: list[Sample]) -> None:
    """Score each sample's full (prompt + response) tokens against a frozen, sglang-served
    teacher model, setting `sample.teacher_log_probs` (or, in full_vocab mode,
    `sample.teacher_hidden_states`) in place. Used for on-policy distillation: the response
    tokens were already sampled from the student's policy during rollout, so we just need
    the teacher's log-probs on those same tokens (--teacher-score-mode sampled_token) or its
    last-layer hidden state at those positions (--teacher-score-mode full_vocab, for
    --loss-type opd_jsd_loss's exact divergence -- the training side reconstructs
    the teacher's full vocab distribution from the hidden state via the teacher's own LM head,
    see orbit/backends/training_utils/teacher_lm_head.py, instead of transmitting the much
    larger full logprob vector over HTTP).

    Sends `max_new_tokens=0` so sglang scores the given tokens instead of generating.
    Deliberately does not reuse `compute_request_payload`: it treats `max_new_tokens <= 0` as
    "no room left to generate" and bails out with a TRUNCATED status, which is the wrong
    semantics here -- a scoring request wants `max_new_tokens=0`.
    """
    # Deferred import: sglang_rollout imports from this module at load time, so importing
    # it back at module level here would be circular.
    from orbit.rollout.sglang_rollout import GenerateState, get_model_url

    state = GenerateState(args)
    semaphore = state.semaphore
    url = get_model_url(args, model_name, "/generate")
    full_vocab = getattr(args, "teacher_score_mode", "sampled_token") == "full_vocab"

    async def _score_one(sample: Sample) -> None:
        if sample.response_length == 0:
            if full_vocab:
                sample.teacher_hidden_states = np.zeros((0, 0), dtype=np.float32)
            else:
                sample.teacher_log_probs = []
            return
        if full_vocab:
            payload = {
                "input_ids": sample.tokens,
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }
            async with semaphore:
                output = await post(url, payload)
            # meta_info["hidden_states"] is wrapped in an extra request-batch dimension
            # (always length 1 here, since each HTTP call scores exactly one sample) --
            # hidden_states[0] is the actual per-position sequence, one vector per token
            # of the full prompt+response input_ids we sent (confirmed empirically: its
            # length matched len(sample.tokens), not response_length).
            outer_hidden_states = output["meta_info"].get("hidden_states") or []
            if len(outer_hidden_states) != 1:
                raise AssertionError(
                    f"expected meta_info['hidden_states'] to have exactly 1 (batch) entry, got "
                    f"{len(outer_hidden_states)} -- sglang's return_hidden_states response shape "
                    "didn't match what compute_teacher_log_probs assumed. See "
                    "orbit/backends/training_utils/teacher_lm_head.py and the full-vocab OPD plan."
                )
            # Reshape against the token count we sent: the trailing dimension is the
            # teacher's hidden size, which we deliberately do not hardcode. A server that
            # returned the wrong number of positions makes this fail outright rather than
            # silently mis-slicing below.
            expected_positions = len(sample.tokens)
            payload_hidden_states = outer_hidden_states[0]
            if isinstance(payload_hidden_states, str):
                raw = pybase64.b64decode(payload_hidden_states.encode("ascii"))
                if len(raw) % (expected_positions * _HIDDEN_STATE_DTYPE.itemsize) != 0:
                    meta_info = output["meta_info"]
                    raise AssertionError(
                        f"teacher hidden_states buffer of {len(raw)} bytes is not a whole number "
                        f"of {_HIDDEN_STATE_DTYPE.name} vectors over {expected_positions} positions "
                        f"(len(sample.tokens)={len(sample.tokens)}, "
                        f"response_length={sample.response_length}, "
                        f"cached_tokens={meta_info.get('cached_tokens')}, "
                        f"cached_tokens_details={meta_info.get('cached_tokens_details')}, "
                        f"prompt_tokens={meta_info.get('prompt_tokens')}, "
                        f"completion_tokens={meta_info.get('completion_tokens')}). sglang's "
                        "return_hidden_states response shape didn't match what "
                        "compute_teacher_log_probs assumed. See "
                        "orbit/backends/training_utils/teacher_lm_head.py and the full-vocab OPD plan."
                    )
                hidden_states = np.frombuffer(raw, dtype=_HIDDEN_STATE_DTYPE).reshape(expected_positions, -1)
            else:
                _warn_legacy_hidden_states_format()
                hidden_states = np.asarray(payload_hidden_states, dtype=_HIDDEN_STATE_DTYPE)
                if hidden_states.ndim != 2 or hidden_states.shape[0] != expected_positions:
                    raise AssertionError(
                        f"teacher hidden_states has shape {hidden_states.shape}, expected "
                        f"({expected_positions}, hidden_size) -- len(sample.tokens)="
                        f"{len(sample.tokens)}, response_length={sample.response_length}. See "
                        "orbit/backends/training_utils/teacher_lm_head.py and the full-vocab OPD plan."
                    )
            # hidden_states[t] is the state after consuming token t, so it is what predicts token t+1
            hidden_start = len(sample.tokens) - sample.response_length - 1
            assert hidden_start >= 0, (
                "full-vocab teacher scoring needs at least one prompt token before the "
                f"response (len(tokens)={len(sample.tokens)}, "
                f"response_length={sample.response_length})"
            )

            sample.teacher_hidden_states = np.array(
                hidden_states[hidden_start : hidden_start + sample.response_length]
            )
            return
        prompt_length = len(sample.tokens) - sample.response_length
        # sglang always returns logprob=None for the first entry of whatever window
        # logprob_start_len opens (a boundary artifact, not a property of that specific
        # token). Starting one token earlier absorbs that None into a throwaway boundary
        # entry (dropped below via [1:]) instead of the first real response token.
        logprob_start_len = max(prompt_length - 1, 0)
        payload = {
            "input_ids": sample.tokens,
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
        }
        async with semaphore:
            output = await post(url, payload)
        input_token_logprobs = output["meta_info"].get("input_token_logprobs") or []
        sample.teacher_log_probs = [item[0] for item in input_token_logprobs[1:]]

    await asyncio.gather(*(_score_one(sample) for sample in samples))

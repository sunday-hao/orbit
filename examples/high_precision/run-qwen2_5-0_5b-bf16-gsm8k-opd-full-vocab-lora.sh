#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 full-vocabulary On-Policy Distillation (OPD) on GSM8K, scored
# against a frozen Qwen2.5-1.5B-Instruct teacher. Unlike the sampled-token OPD launcher
# (run-qwen2_5-0_5b-bf16-gsm8k-opd-lora.sh), the teacher returns its full-vocabulary
# distribution at every response position (--teacher-score-mode full_vocab) and the student
# is trained with an exact KL divergence (--loss-type opd_full_vocab_loss) instead of the
# REINFORCE-style teacher_log_prob - student_log_prob advantage. Self-contained launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen25_05b_bf16_gsm8k_megatron_opd_full_vocab_lora
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ROOT}/train_opd.py"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the student Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the student Megatron torch_dist checkpoint path}"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_gsm8k_opd_full_vocab_lora"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a GSM8K training data path (.jsonl or .parquet)}"
: "${TEST_JSONL:?set TEST_JSONL to a GSM8K eval data path (.jsonl or .parquet), or set DISABLE_EVAL=1}"

# Teacher checkpoint, served frozen via SGLang -- never converted to Megatron, never
# trained. MUST be same-family (same tokenizer/vocab) as the student checkpoint above.
: "${OPD_TEACHER_CKPT:?set OPD_TEACHER_CKPT to the teacher Hugging Face checkpoint path}"

# === Resources ===
# Single GPU, --colocate: actor training, student rollout serving, and teacher serving all
# time-share this one GPU via the offload/onload dance (see orbit/ray/teacher.py and the
# create_opd_placement_groups() colocate branch in orbit/ray/placement_group.py).
GPUS_PER_NODE=1
RAY_NUM_CPUS=64

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"

# `wc -l` undercounts .parquet files (binary) -- count rows with pyarrow instead.
count_rows() {
    case "$1" in
        *.parquet) python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$1').metadata.num_rows)" ;;
        *) wc -l < "$1" ;;
    esac
}
TRAIN_ROWS=${TRAIN_ROWS:-$(count_rows "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === OPD teacher sglang_config ===
# The frozen teacher is a second, update_weights=false model behind its own router, served
# on the "teacher" placement group slice (create_opd_placement_groups() in
# orbit/ray/placement_group.py). Regenerated each run since it embeds OPD_TEACHER_CKPT.
OPD_SGLANG_CONFIG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_sglang_config.yaml"
mkdir -p "$(dirname "${OPD_SGLANG_CONFIG}")"
cat > "${OPD_SGLANG_CONFIG}" <<EOF
sglang:
  - name: default
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 1
  - name: teacher
    model_path: "${OPD_TEACHER_CKPT}"
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          mem_fraction_static: 0.25
          # --teacher-score-mode full_vocab needs the teacher's last-layer hidden states
          # (return_hidden_states=True per-request) to reconstruct its full vocab
          # distribution on the training side -- this is the server-startup flag that
          # allows that.
          enable_return_hidden_states: true
          # A radix-cache hit skips the forward pass for the matched prefix, so no hidden
          # state gets captured for those positions -- confirmed on a real run (a
          # per-request cache-salt didn't reliably avoid this). Disabling the cache
          # entirely guarantees every position gets a fresh hidden state; the teacher is
          # only ever used for scoring here, so losing prefix-cache speedups doesn't cost
          # much.
          disable_radix_cache: true
          # Root cause of hidden_states truncation, confirmed against sglang's own source
          # (scheduler_output_processor_mixin.py): hidden-state capture lives inside
          # `if req.is_chunked <= 0:`, so only the LAST chunk of a chunked-prefill request
          # ever gets its hidden states appended -- every earlier chunk is silently
          # dropped. -1 disables chunked prefill entirely (not "a very large chunk size" --
          # sglang maps -1 to rem_chunk_tokens=None internally, its "chunking disabled"
          # state), so a scoring request's prefill is never split into chunks regardless of
          # length, rather than relying on a fixed size staying ahead of
          # --rollout-max-response-len.
          chunked_prefill_size: -1
EOF

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 200
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

# --teacher-score-mode full_vocab makes compute_teacher_log_probs request the teacher's
# last-layer hidden state at every response position (return_hidden_states=True), instead
# of just the sampled token's log-prob. --teacher-hf-checkpoint lets the training side
# reconstruct the teacher's full vocab distribution from that hidden state via the
# teacher's own LM head (orbit/backends/training_utils/teacher_lm_head.py) -- far cheaper
# than shipping a vocab-sized logprob vector per token over HTTP. --loss-type
# opd_full_vocab_loss then computes the exact KL(student || teacher) directly from logits
# -- no advantage/returns pipeline, hence --disable-compute-advantages-and-returns.
RL_ARGS=(
    --advantage-estimator on_policy_distillation
    --teacher-model-name teacher
    --teacher-score-mode full_vocab
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --disable-compute-advantages-and-returns
)

LOSS_ARGS=(
    --loss-type opd_full_vocab_forward_kl_loss
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

# opd_full_vocab_loss requires tensor_model_parallel_size == 1 and context_parallel_size ==
# 1 (asserted in orbit/utils/arguments.py) -- fine here since these are already 1.
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data math "${TEST_JSONL}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 1024
    --eval-top-k 1
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --num-gpus-per-node 1
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.3
    --rollout-num-gpus 0
    --teacher-num-gpus 1
    --sglang-config "${OPD_SGLANG_CONFIG}"
    --sglang-max-running-requests 1024
    --router-disable-circuit-breaker
    # flashinfer, not fa3 -- SGLang has no separate "fa2" backend name, flashinfer is its
    # own FA2-equivalent kernel.
    --sglang-attention-backend fa3
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=(
    --log-passrate
)

PEFT_ARGS=(
    --peft-method lora
    --peft-variant standard
    --lora-rank 32
    --lora-alpha 64
    --lora-dropout 0.0
    --target-modules all-linear
)
source "${ORBIT_ROOT}/scripts/lib/launcher.sh"

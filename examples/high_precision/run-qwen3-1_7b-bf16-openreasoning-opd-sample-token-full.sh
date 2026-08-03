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
LAUNCHER_NAME=run_qwen3_17b_bf16_openreasoning100k_megatron_sampled_token_opd_full
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ROOT}/train_opd.py"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the student Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the student Megatron torch_dist checkpoint path}"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen3-1.7B_4B_Instruct2507_openreasoning100k_sampled_token_opd_full"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a GSM8K training data path (.jsonl or .parquet)}"
AIME24_PATH="${ORBIT_ROOT}/data/aime24/test.parquet"
AIME25_PATH="${ORBIT_ROOT}/data/aime25/test.parquet"
HMMT25_PATH="${ORBIT_ROOT}/data/hmmt25/test.parquet"

# Teacher checkpoint, served frozen via SGLang -- never converted to Megatron, never
# trained. MUST be same-family (same tokenizer/vocab) as the student checkpoint above.
: "${OPD_TEACHER_CKPT:?set OPD_TEACHER_CKPT to the teacher Hugging Face checkpoint path}"

# === Resources ===
# Single GPU, --colocate: actor training, student rollout serving, and teacher serving all
# time-share this one GPU via the offload/onload dance (see orbit/ray/teacher.py and the
# create_opd_placement_groups() colocate branch in orbit/ray/placement_group.py).
GPUS_PER_NODE=4
RAY_NUM_CPUS=64

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
#TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

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
        num_gpus: 4
  - name: teacher
    model_path: "${OPD_TEACHER_CKPT}"
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
        overrides:
          mem_fraction_static: 0.3
          max_prefill_tokens: 4096
          max_running_requests: 8
EOF

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 10
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key question
    --label-key answer
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style cosine
    --min-lr 5e-7
    --lr-warmup-fraction 0.1
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)


RL_ARGS=(
    --advantage-estimator on_policy_distillation
    --teacher-model-name teacher
    --teacher-score-mode sampled_token
)

LOSS_ARGS=(
    --loss-type policy_loss
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)


PERF_ARGS=(
    --tensor-model-parallel-size 4
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
    --skip-eval-before-train
    --eval-prompt-data aime24 "${AIME24_PATH}" aime25 "${AIME25_PATH}" hmmt25 "${HMMT25_PATH}"
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 8192
    --eval-top-k -1
    --eval-top-p 0.95
    --eval-temperature 1.0
    --eval-pass-k-values 1 8 16
)

SGLANG_ARGS=(
    --num-gpus-per-node 4
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.25
    --sglang-server-concurrency 4 ##for memory saving, we set it to 4, but it can be set to 8 for better throughput
    --rollout-num-gpus 0
    --teacher-num-gpus 4
    --sglang-config "${OPD_SGLANG_CONFIG}"
    --sglang-max-running-requests 512
    --router-disable-circuit-breaker
    # flashinfer, or fa3
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
    --peft-method none
)
source "${ORBIT_ROOT}/scripts/lib/launcher.sh"

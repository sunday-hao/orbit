#!/usr/bin/env bash
# M6 -- uncentered sampled-token RKL estimator on refreshed student rollouts,
# full fine-tuning.
#
# One-shot: trains 300 constant-LR steps, then scores every saved checkpoint on
# aime24/aime25/amc23/math500. Self-contained -- no shared pipeline file.
#
# Ported from trl/recipes/unified/300step_constant/m6_opd_st_non_thinking.yaml.
# Held constant across M4/M5/M6 and across full/lora/oft so the nine cells stay
# comparable: 300 optimizer steps, global batch 256, constant LR 5e-6 with no warmup
# or decay, weight decay 0, grad-norm clip 1.0, seed 42, one generation per prompt at
# temperature 0.7 / top-p 1.0, 8192-token completions, non-thinking chat template,
# checkpoint every 20 steps (15 in total).
#
# Objective: the single-sample estimator of the same reverse KL M5 computes
# exactly. The teacher scores only the token the student sampled.
#
# Mapping from the recipe: distillation_objective=iw_opd with iw_opd_gamma=0.0 and
# iw_opd_center_top_k=0 reduces IWOPDTrainer to a_opd = teacher_logp - student_logp
# with unit weights, optimised as loss = -student_logp * a_opd. That is exactly
# orbit's --advantage-estimator on_policy_distillation, and --force-on-policy-ratio
# pins the PPO ratio to 1 so the surrogate reduces to the same REINFORCE form.
# Advantages are left unnormalised (--normalize-advantages defaults off), matching
# the recipe. The recipe's reward_scale: 0.1 has no consumer in the iw_opd trainer
# and is not carried over.
#
# Temperature: no teacher-side scaling here, and none is needed -- the teacher
# contributes a single sampled-token log-prob, which has no normalizer to re-temper
# against. (The full-vocab launchers do scale the teacher by --rollout-temperature,
# because there the whole teacher distribution is reconstructed trainer-side and the
# two distributions have to be compared on the same footing.)
#
# Checkpoints are full-finetune torch_dist shards. The eval stage converts each to
# dense HF weights with tools/convert_torch_dist_to_hf.py, using BASE_MODEL=HF_CKPT.
#
# The recipe runs DeepSpeed ZeRO-3 on eight GPUs; orbit is Megatron and this runs on
# four, so the parallelism split below (TP2 x DP2, colocated rollout, 2-GPU managed
# teacher) is an orbit-side choice rather than a transcription. Global batch 256 is
# preserved, so the optimization trajectory is unchanged; only throughput differs.
#
#   HF_CKPT=/path/to/hf/Qwen3-1.7B \
#   MEGATRON_LOAD=/path/to/megatron/Qwen3-1.7B \
#   OPD_TEACHER_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507 \
#   TRAIN_JSONL=/path/to/opd_mixed_100k_shard_balanced \
#   EVALCHEMY_ROOT=/path/to/evalchemy \
#       bash examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full.sh
#
# RUN_TRAIN=0 evaluates an existing run; RUN_EVAL=0 trains only.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=unified_300step_constant_m6_opd_st_full_non_thinking
WANDB_PROJECT=${WANDB_PROJECT:-orbit-adapt}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths, shared by both stages ===
# Cluster defaults; override any of them in the environment.
HF_CKPT="${HF_CKPT:-/mnt/L202500430/orbit/data/hf_ckpts/Qwen3-1.7B}"
MEGATRON_LOAD="${MEGATRON_LOAD:-/mnt/L202500430/orbit/data/megatron_ckpts/Qwen3-1.7B}"
OPD_TEACHER_CKPT="${OPD_TEACHER_CKPT:-/mnt/L202500430/orbit/data/hf_ckpts/Qwen3-4B-Instruct-2507}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/L202500430/orbit/data/openreasoning_mixed_100k/train_qa.parquet}"
EVALCHEMY_ROOT="${EVALCHEMY_ROOT:-/mnt/L202500430/evalchemy}"
# Grading runs in its own venv: lm_eval is not in the Orbit environment.
RUNNER_PYTHON_BIN="${RUNNER_PYTHON_BIN:-/mnt/L202500430/.venv-opd-eval/bin/python}"

# Fail here with a readable message rather than deep inside Megatron/SGLang.
for _p in "${HF_CKPT}" "${MEGATRON_LOAD}" "${OPD_TEACHER_CKPT}" "${TRAIN_JSONL}"; do
    if [ ! -e "${_p}" ]; then
        echo "[$(basename "${BASH_SOURCE[0]}")] ERROR: path does not exist: ${_p}" >&2
        exit 1
    fi
done
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/${LAUNCHER_NAME}}"

# In-training eval is off (the recipe sets eval_strategy: "no"); scoring is the
# evalchemy sweep in stage 2 below.
DISABLE_EVAL=1

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
# Fail now, not after a multi-hour train, if the eval stage could never work.
if [ "${RUN_EVAL}" = "1" ]; then
    if [ ! -d "${EVALCHEMY_ROOT}/eval/chat_benchmarks" ]; then
        echo "[m6_opd_st_full] ERROR: EVALCHEMY_ROOT='${EVALCHEMY_ROOT}' has no eval/chat_benchmarks/" >&2
        exit 1
    fi
fi

# === Resources ===
# Four GPUs. --colocate: actor training, student rollout, and the managed teacher
# time-share them. The recipe's eight-GPU shape (per_device 1 x grad_accum 32 x 8)
# existed only to reach global batch 256; --global-batch-size is a global quantity in
# orbit, so 256 is preserved here with more accumulation per GPU. TP2 x DP2.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${GPUS_PER_NODE}}"
OPD_TEACHER_NUM_GPUS="${OPD_TEACHER_NUM_GPUS:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
# max_steps 300 at global batch 256. rollout_batch_size x n_samples_per_prompt equals
# global_batch_size, so one rollout is exactly one optimizer step and NUM_ROLLOUT is
# the recipe's step count. 300 x 256 = 76,800 prompts, no epoch wrap.
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-20}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key "${INPUT_KEY:-question}"
    --label-key "${LABEL_KEY:-answer}"
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    # --rollout-max-prompt-len 1024
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    # Required transport for a served teacher: reward_func POSTs the sampled sequence
    # for scoring, post_process puts the result on the sample.
    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process
)

# Constant 5e-6, no warmup, no decay.
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-5e-6}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --clip-grad 1.0
)

# The frozen teacher is served by this job: orbit launches it as an extra sglang model
# entry with update_weights=false and the scoring-correctness server flags baked in
# (orbit/ray/rollout.py::_teacher_server_overrides).
RL_ARGS=(
    --opd-type sglang
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --opd-serve-teacher
    --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS}"
    --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION:-0.35}"
    --opd-teacher-max-running-requests "${OPD_TEACHER_MAX_RUNNING_REQUESTS:-32}"
    --opd-teacher-max-prefill-tokens "${OPD_TEACHER_MAX_PREFILL_TOKENS:-4096}"
    --advantage-estimator on_policy_distillation
    --teacher-score-mode sampled_token
    --force-on-policy-ratio
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
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
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

# Emptied by validate_eval_args because DISABLE_EVAL=1 above.
EVAL_ARGS=()

SGLANG_ARGS=(
    --num-gpus-per-node "${GPUS_PER_NODE}"
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.25}"
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-512}"
    --router-disable-circuit-breaker
    # fa3 is rejected on B200/SM100 by the pinned SGLang; use triton there.
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-fa3}"
    --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND:-flashinfer}"
)

MISC_ARGS=(
    --seed 42
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

DEBUG_ARGS=( --log-passrate )

PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"

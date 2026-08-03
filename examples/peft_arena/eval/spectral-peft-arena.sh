#!/usr/bin/env bash
# Run PEFT-Arena's weights-only geometry diagnostics against Orbit-trained PEFT
# adapter checkpoints, in one pass:
#
#   spectral_analysis.py      dual-view spectral analysis, ΔW spectrum, SVA
#   compare_model_norms.py    L1/L2 base-vs-tuned norm comparison (RUN_NORMS=1)
#   spectral_rank_metrics.py  Frobenius norm, stable rank, energy-effective
#                             rank, energy coverage@k -- derived from the
#                             spectrum written above, no second SVD
#
# Two modes, picked by which path variable you set:
#
#   SAVE_DIR=orbit_ckpts/<run>          sweep every iter_*/adapter serially and
#                                       roll the run up into one CSV per metric
#                                       family, with `iter` as the first column
#   ITER_DIR=.../iter_NNNNNNN/adapter   analyze exactly one checkpoint
#
# Sweep mode is resume-safe: iters that already have spectral_summary.json are
# skipped, so re-running after a crash or after new checkpoints land is cheap.
#
# Common overrides (defaults shown):
#   PEFT_ARENA_ROOT=${ORBIT_ROOT}/../PEFT-Arena
#   BASE_MODEL=<adapter_config.json base_model_name_or_path>
#   EVAL_RESULTS_ROOT=${ORBIT_ROOT}/eval_results     (sweep mode)
#   OUTPUT_DIR=${EVAL_RESULTS_ROOT}/<run>/<iter>/spectral
#   MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
#   LAYERS=                      (empty = every layer)
#   SMOOTHNESS_WINDOW=5
#   RUN_NORMS=1                  (also run compare_model_norms.py)
#   COVERAGE_RANKS=8,16,32,64    (k values for energy coverage@k)
#   KEEP_GOING=0                 (sweep aborts on the first failing checkpoint
#                                 and prints its log tail; 1 = push through)
#   PYTHON_BIN=<orbit workspace venv python if present, else `python` on PATH>
#
# Examples:
#   SAVE_DIR=$(pwd)/orbit_ckpts/<run> COVERAGE_RANKS=1,2,4,8,16,24 \
#       bash examples/peft_arena/eval/spectral-peft-arena.sh
#   ITER_DIR=$(pwd)/orbit_ckpts/<run>/iter_0000050/adapter \
#       bash examples/peft_arena/eval/spectral-peft-arena.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# examples/peft_arena/eval -> repo root is three levels up, not two.
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=../../scripts/lib/tool_env.sh
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"

if [ -z "${SAVE_DIR:-}" ] && [ -z "${ITER_DIR:-}" ]; then
    echo "[spectral-peft-arena] ERROR: set SAVE_DIR (sweep a run) or ITER_DIR (one checkpoint)." >&2
    exit 1
fi

PEFT_ARENA_ROOT="${PEFT_ARENA_ROOT:-$(cd -- "${ORBIT_ROOT}/.." && pwd)/PEFT-Arena}"
SPECTRAL_SCRIPT="${PEFT_ARENA_ROOT}/tools/spectral_analysis.py"
NORMS_SCRIPT="${PEFT_ARENA_ROOT}/tools/compare_model_norms.py"
RANK_SCRIPT="${ORBIT_ROOT}/tools/spectral_rank_metrics.py"
SUMMARY_SCRIPT="${ORBIT_ROOT}/tools/summarize_spectral_results.py"
# Upstream tools run through this shim so ../PEFT-Arena stays an unpatched
# clone; it neutralizes peft's TorchAO probe, which raises (rather than
# returning False) when the installed torchao predates the peft requirement.
RUN_TOOL="${ORBIT_ROOT}/tools/run_peft_arena_tool.py"

if [ ! -f "${SPECTRAL_SCRIPT}" ]; then
    echo "[spectral-peft-arena] ERROR: PEFT_ARENA_ROOT='${PEFT_ARENA_ROOT}' does not contain tools/spectral_analysis.py." >&2
    echo "  These diagnostics are NOT in the vendored snapshot under examples/peft_arena/backend/." >&2
    echo "  Clone the full repo next to orbit:" >&2
    echo "    git clone --depth 1 --filter=blob:none https://github.com/Sphere-AI-Lab/PEFT-Arena.git ${ORBIT_ROOT}/../PEFT-Arena" >&2
    exit 1
fi

# Prefer the orbit workspace venv when it is actually there, otherwise fall back
# to whatever `python` the caller has active -- on the cluster that is the
# already-activated training env, and hard-coding the workspace path made the
# script fail with a misleading "no CUDA device" message when it did not exist.
if [ -z "${PYTHON_BIN:-}" ]; then
    _orbit_venv_python="${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace/.venv/bin/python"
    if [ -x "${_orbit_venv_python}" ]; then
        PYTHON_BIN="${_orbit_venv_python}"
    else
        PYTHON_BIN="$(command -v python || command -v python3 || true)"
    fi
fi
if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[spectral-peft-arena] ERROR: no usable python found (PYTHON_BIN='${PYTHON_BIN}')." >&2
    echo "  Activate the training env, or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi
echo "[spectral-peft-arena] Python: ${PYTHON_BIN}"

# --- Knobs shared by both modes --------------------------------------------
# Default to the seven attention/MLP projections. Leaving MODULES empty makes
# get_linear_layer_pairs() pick up every 2-D tensor, which includes
# embed_tokens/lm_head -- a 151936x2048 SVD for Qwen3 that dominates runtime and
# is not what the dual-view analysis is about.
MODULES="${MODULES-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LAYERS="${LAYERS-}"
SMOOTHNESS_WINDOW="${SMOOTHNESS_WINDOW:-5}"
RUN_NORMS="${RUN_NORMS:-1}"
COVERAGE_RANKS="${COVERAGE_RANKS:-8,16,32,64}"

format_duration() {
    local seconds="$1"
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    if [ "${hours}" -gt 0 ]; then
        printf '%dh%02dm%02ds' "${hours}" "${minutes}" "$((seconds % 60))"
    else
        printf '%dm%02ds' "${minutes}" "$((seconds % 60))"
    fi
}

format_eta() {
    local done_count="$1" total_count="$2" started_at="$3" now="$4"
    if [ "${done_count}" -le 0 ]; then
        printf 'unknown'
        return
    fi
    printf '%s' "$(format_duration $(( (now - started_at) * (total_count - done_count) / done_count )))"
}

# Sets RUN_NAME/ITER_NAME for any of the analyzable layouts: a PEFT
# iter_NNNNNNN/adapter, a full-finetune iter_NNNNNNN/hf, or an iter_NNNNNNN dir
# that is itself an HF export.
resolve_run_and_iter() {
    local dir="$1"
    if [[ "$(basename "${dir}")" == iter_* ]]; then
        ITER_NAME="$(basename "${dir}")"
        RUN_NAME="$(basename "$(dirname "${dir}")")"
    else
        ITER_NAME="$(basename "$(dirname "${dir}")")"
        RUN_NAME="$(basename "$(dirname "$(dirname "${dir}")")")"
    fi
}

# ===========================================================================
# Sweep mode
# ===========================================================================
if [ -n "${SAVE_DIR:-}" ] && [ -z "${ITER_DIR:-}" ]; then
    SAVE_DIR="$(cd -- "${SAVE_DIR}" && pwd)"
    RUN_NAME="$(basename "${SAVE_DIR}")"
    EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${ORBIT_ROOT}/eval_results}"
    EVAL_RESULTS="${EVAL_RESULTS_ROOT}/${RUN_NAME}"
    LOG_DIR="${LOG_DIR:-${ORBIT_ROOT}/logs}"
    mkdir -p "${EVAL_RESULTS}" "${LOG_DIR}"

    # Re-entering this script per checkpoint keeps one code path for the actual
    # analysis. Export the *resolved* knobs so the child inherits them already
    # defaulted -- that also removes the unset-vs-empty MODULES subtlety.
    export PEFT_ARENA_ROOT PYTHON_BIN MODULES LAYERS SMOOTHNESS_WINDOW RUN_NORMS COVERAGE_RANKS
    [ -n "${BASE_MODEL:-}" ] && export BASE_MODEL

    # Three per-iter layouts are analyzable, because spectral_analysis.py takes
    # either a PEFT adapter dir or a plain HF model dir:
    #   iter_NNNNNNN/adapter/   PEFT run (save_checkpoint_with_peft)
    #   iter_NNNNNNN/hf/        full finetune, already exported
    #   iter_NNNNNNN/           full finetune, exported in place
    # A full-finetune run that was never exported holds Megatron torch_dist
    # shards instead; those are reported at the end with the conversion command
    # rather than silently counted as "0 checkpoints found".
    declare -a analyze_dirs=()
    declare -a unconverted_iters=()
    for iter_dir in "${SAVE_DIR}"/iter_*; do
        [ -d "${iter_dir}" ] || continue
        if [ -f "${iter_dir}/adapter/adapter_model.safetensors" ] && [ -f "${iter_dir}/adapter/adapter_config.json" ]; then
            analyze_dirs+=("${iter_dir}/adapter")
        elif [ -f "${iter_dir}/hf/config.json" ]; then
            analyze_dirs+=("${iter_dir}/hf")
        elif [ -f "${iter_dir}/config.json" ]; then
            analyze_dirs+=("${iter_dir}")
        else
            unconverted_iters+=("${iter_dir}")
        fi
    done

    total_count="${#analyze_dirs[@]}"
    echo "[spectral-peft-arena] sweep: ${total_count} analyzable checkpoints under ${SAVE_DIR}"
    echo "[spectral-peft-arena] results under ${EVAL_RESULTS}"
    echo "[spectral-peft-arena] modules='${MODULES}' layers='${LAYERS:-all}' coverage_ranks='${COVERAGE_RANKS}'"
    if [ "${#unconverted_iters[@]}" -gt 0 ]; then
        echo "[spectral-peft-arena] WARNING: ${#unconverted_iters[@]} iter dir(s) contain neither a PEFT adapter nor HF weights." >&2
        echo "  A full-finetune run saves Megatron torch_dist shards, which spectral_analysis.py cannot read." >&2
        echo "  Export each one to HF first, then re-run this sweep:" >&2
        echo "    ${PYTHON_BIN} ${ORBIT_ROOT}/tools/convert_torch_dist_to_hf.py \\" >&2
        echo "        --input-dir ${unconverted_iters[0]} --output-dir ${unconverted_iters[0]}/hf" >&2
    fi

    if [ "${total_count}" -eq 0 ]; then
        echo "[spectral-peft-arena] ERROR: no analyzable checkpoint under ${SAVE_DIR}" >&2
        exit 1
    fi

    started_at="$(date +%s)"
    done_count=0
    failed_count=0

    for analyze_dir in "${analyze_dirs[@]}"; do
        resolve_run_and_iter "${analyze_dir}"
        iter_name="${ITER_NAME}"
        done_count=$((done_count + 1))
        iter_output_dir="${EVAL_RESULTS}/${iter_name}/spectral"

        if [ -f "${iter_output_dir}/spectral_summary.json" ]; then
            now="$(date +%s)"
            echo "[spectral-peft-arena] $(date +%H:%M:%S) ${iter_name}: already analyzed, skipping (${done_count}/${total_count}, ETA $(format_eta "${done_count}" "${total_count}" "${started_at}" "${now}"))"
            continue
        fi

        echo "[spectral-peft-arena] $(date +%H:%M:%S) analyzing ${iter_name} (${done_count}/${total_count})"
        iter_started_at="$(date +%s)"
        log="${LOG_DIR}/spectral_${RUN_NAME}_${iter_name}.log"
        if ITER_DIR="${analyze_dir}" OUTPUT_DIR="${iter_output_dir}" \
           bash "${BASH_SOURCE[0]}" > "${log}" 2>&1; then
            status="done"
        else
            status="FAIL"
            failed_count=$((failed_count + 1))
        fi

        now="$(date +%s)"
        echo "[spectral-peft-arena] $(date +%H:%M:%S) ${status} ${iter_name} duration=$(format_duration "$((now - iter_started_at))") (${done_count}/${total_count}, ETA $(format_eta "${done_count}" "${total_count}" "${started_at}" "${now}"), log: ${log})"

        # A failure here is almost always systematic (wrong base model, missing
        # dep, unreadable adapter), so grinding through the remaining
        # checkpoints just reproduces it. Surface the cause and stop.
        if [ "${status}" = "FAIL" ]; then
            echo "[spectral-peft-arena] ---- tail of ${log} ----" >&2
            tail -n 25 "${log}" >&2
            echo "[spectral-peft-arena] ---- end of log ----" >&2
            if [ "${KEEP_GOING:-0}" != "1" ]; then
                echo "[spectral-peft-arena] stopping after the first failure; set KEEP_GOING=1 to run the whole sweep anyway." >&2
                exit 1
            fi
        fi
    done

    # Run-level rollups: one row per checkpoint, `iter` first -- plot directly.
    "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --run-dir "${EVAL_RESULTS}" >/dev/null
    "${PYTHON_BIN}" "${RANK_SCRIPT}" --run-dir "${EVAL_RESULTS}" --coverage-ranks "${COVERAGE_RANKS}" >/dev/null

    echo "[spectral-peft-arena] sweep done."
    echo "  rank metrics vs iter:  ${EVAL_RESULTS}/rank_metrics_summary.csv"
    echo "  rank metrics per layer: ${EVAL_RESULTS}/rank_metrics_per_layer.csv"
    echo "  dual-view vs iter:     ${EVAL_RESULTS}/spectral_summary.csv"
    echo "  dual-view per layer:   ${EVAL_RESULTS}/spectral_per_layer.csv"

    if [ "${failed_count}" -gt 0 ]; then
        echo "[spectral-peft-arena] ${failed_count} checkpoint(s) failed" >&2
        exit 1
    fi
    exit 0
fi

# ===========================================================================
# Single-checkpoint mode
# ===========================================================================
ITER_DIR="$(cd -- "${ITER_DIR}" && pwd)"

# --- Resolve the base model ------------------------------------------------
# spectral_analysis.py takes base and finetuned paths separately and never reads
# base_model_name_or_path itself, so a stale path inside adapter_config.json is
# recoverable here by exporting BASE_MODEL -- unlike the math wrapper.
if [ -z "${BASE_MODEL:-}" ]; then
    if [ ! -f "${ITER_DIR}/adapter_config.json" ]; then
        echo "[spectral-peft-arena] ERROR: ${ITER_DIR} has no adapter_config.json and BASE_MODEL is unset." >&2
        echo "  A full-finetune HF checkpoint records no base model, so ΔW = W_ft - W_pre needs" >&2
        echo "  BASE_MODEL=/path/to/hf_models/<NAME> pointing at the pre-trained weights." >&2
        exit 1
    fi
    BASE_MODEL="$(${PYTHON_BIN} - <<EOF
from peft import PeftConfig
import sys
cfg = PeftConfig.from_pretrained("${ITER_DIR}")
base = cfg.base_model_name_or_path or ""
if not base:
    sys.exit("adapter_config.json missing base_model_name_or_path; set BASE_MODEL explicitly")
print(base)
EOF
)"
fi

if [ ! -d "${BASE_MODEL}" ]; then
    echo "[spectral-peft-arena] ERROR: BASE_MODEL '${BASE_MODEL}' does not exist on disk." >&2
    echo "  Export BASE_MODEL=/path/to/hf_models/<NAME> to override the checkpoint's recorded path." >&2
    exit 1
fi
echo "[spectral-peft-arena] Base model: ${BASE_MODEL}"
echo "[spectral-peft-arena] Checkpoint: ${ITER_DIR}"

resolve_run_and_iter "${ITER_DIR}"
EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${ORBIT_ROOT}/eval_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/${RUN_NAME}/${ITER_NAME}/spectral}"
mkdir -p "${OUTPUT_DIR}"
echo "[spectral-peft-arena] Output dir: ${OUTPUT_DIR}"

SPECTRAL_ARGS=(
    --base_model "${BASE_MODEL}"
    --finetuned_model "${ITER_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --smoothness_window "${SMOOTHNESS_WINDOW}"
)
[ -n "${MODULES}" ] && SPECTRAL_ARGS+=(--modules "${MODULES}")
[ -n "${LAYERS}" ] && SPECTRAL_ARGS+=(--layers "${LAYERS}")

echo "[spectral-peft-arena] Running spectral_analysis.py (modules='${MODULES}' layers='${LAYERS:-all}')"
${PYTHON_BIN} "${RUN_TOOL}" "${SPECTRAL_SCRIPT}" "${SPECTRAL_ARGS[@]}"

if [ "${RUN_NORMS}" = "1" ]; then
    echo "[spectral-peft-arena] Running compare_model_norms.py"
    ${PYTHON_BIN} "${RUN_TOOL}" "${NORMS_SCRIPT}" \
        --base-model "${BASE_MODEL}" \
        --sft-model "${ITER_DIR}" \
        --output "${OUTPUT_DIR}/norms"
fi

# Frobenius norm, stable rank, energy-effective rank and coverage@k are all
# functions of the ΔW spectrum that spectral_analysis.py just wrote into the
# per-layer .pt files, so this is a cheap post-pass over OUTPUT_DIR rather than
# a second SVD.
echo "[spectral-peft-arena] Deriving rank metrics (coverage ranks: ${COVERAGE_RANKS})"
${PYTHON_BIN} "${RANK_SCRIPT}" \
    --spectral-dir "${OUTPUT_DIR}" \
    --coverage-ranks "${COVERAGE_RANKS}"

echo "[spectral-peft-arena] Done."
echo "  spectral summary: ${OUTPUT_DIR}/spectral_summary.json"
echo "  rank metrics:     ${OUTPUT_DIR}/rank_metrics.csv"
# Not `[ ... ] && echo` -- as the script's last command a false test would make
# the whole wrapper exit 1 and the sweep would count it as a failure.
if [ "${RUN_NORMS}" = "1" ]; then
    echo "  norms:            ${OUTPUT_DIR}/norms.csv"
fi

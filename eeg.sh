#!/bin/bash
#SBATCH --job-name=spdmlr_tsmnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D33_186
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

ORIGINAL_ARGS=("$@")

DATASET_CFG="stieger2021"
EVALUATION_CFG="inter-subject+uda"
FRAMEWORK="tsmnet"
SEED_LIST="0,1,2,3,4"
N_JOBS=8
MAX_BATCH_SIZE=20
METRICS="lecm,ecm"
USE_LP="true"
USE_LOGM="true"
N_PROJ=150
SWD_POWER=1.0
LAMBDA1=0.75
LAMBDA2=0.75
LR=1e-3
LOSS_LR=5e-4
MAX_RUNTIME_HOURS=23
AUTO_RESUBMIT=1
RESUME=1
FORCE_SPDSW=0
OUTER_PARALLEL=1

OPENNEURO_DS004745_DIR=/leonardo_scratch/fast/EUHPC_D33_186/openneuro/ds004745
OPENNEURO_DS004306_DIR=/leonardo_scratch/fast/EUHPC_D33_186/openneuro/ds004306
HF_ALLJOINED_05_125_CACHE_DIR=/leonardo_scratch/fast/EUHPC_D33_186/hf_cache/datasets/Alljoined___05_125
MOABB_ROOT_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data
MOABB_STIEGER2021_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-Stieger2021-data
MOABB_LEE2019_MI_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-lee2019-mi-data
MOABB_OFNER2017_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-upperlimb-data
MOABB_BNCI2014_001_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-bnci-data/database/data-sets/001-2014
MOABB_BNCI2015_001_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-bnci-data/database/data-sets/001-2015
MOABB_BNCI2025_001_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-bnci-data/~bci/database/001-2025
MOABB_BNCI2025_002_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-bnci-data/database/data-sets/002-2025
MOABB_BEETL2021_A_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-Beetl2021-A-data
MOABB_BEETL2021_B_DIR=/leonardo_scratch/fast/EUHPC_D33_186/eeg_datasets/moabb_mne_data/MNE-Beetl2021-B-data
EEG_DATA_ROOT="${MOABB_ROOT_DIR}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--datasets|--datsets) DATASET_CFG="$2"; shift 2 ;;
    --evaluation) EVALUATION_CFG="$2"; shift 2 ;;
    --session) EVALUATION_CFG="inter-session+uda"; shift ;;
    --subject) EVALUATION_CFG="inter-subject+uda"; shift ;;
    --framework) FRAMEWORK="$2"; shift 2 ;;
    --data-root) EEG_DATA_ROOT="$2"; shift 2 ;;
    --seeds) SEED_LIST="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
    --metrics|--distance|--distances) METRICS="$2"; shift 2 ;;
    --use-lp|--lp) USE_LP="$2"; shift 2 ;;
    --use-logm|--logm) USE_LOGM="$2"; shift 2 ;;
    --n-proj) N_PROJ="$2"; shift 2 ;;
    --swd-power) SWD_POWER="$2"; shift 2 ;;
    --lambda1) LAMBDA1="$2"; shift 2 ;;
    --lambda2) LAMBDA2="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --loss-lr) LOSS_LR="$2"; shift 2 ;;
    --max-runtime-hours) MAX_RUNTIME_HOURS="$2"; shift 2 ;;
    --outer-parallel) OUTER_PARALLEL="$2"; shift 2 ;;
    --auto-resubmit) AUTO_RESUBMIT=1; shift ;;
    --no-auto-resubmit) AUTO_RESUBMIT=0; shift ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --spdsw) FORCE_SPDSW=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${FORCE_SPDSW}" -eq 1 ]]; then
  USE_LP="false"
  USE_LOGM="true"
fi
if [[ "${METRICS,,}" == "all" ]]; then
  METRICS="ecm,lecm,olm,lsm"
fi

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
module load cuda/12.1

VENV_CANDIDATES=(
  "$WORK/venvs/corsw/bin/activate"
  "/leonardo_work/EUHPC_D33_186/venvs/corsw/bin/activate"
  "$WORK/icmlrick/bin/activate"
  "/leonardo_work/EUHPC_D33_186/icmlrick/bin/activate"
)
FOUND_VENV=""
for v in "${VENV_CANDIDATES[@]}"; do
  if [[ -f "$v" ]]; then
    FOUND_VENV="$v"
    break
  fi
done
if [[ -z "$FOUND_VENV" ]]; then
  echo "[FATAL] venv activate not found." >&2
  exit 1
fi
source "$FOUND_VENV"

PROJECT_CANDIDATES=(
  "${PROJECT_DIR:-}"
  "${SLURM_SUBMIT_DIR:-}"
  "$PWD"
)
FOUND_PROJECT=""
for p in "${PROJECT_CANDIDATES[@]}"; do
  if [[ -n "$p" && -f "$p/TSMNet-MLR.py" ]]; then
    FOUND_PROJECT="$p"
    break
  fi
done
if [[ -z "$FOUND_PROJECT" ]]; then
  echo "[FATAL] project dir not found." >&2
  exit 1
fi
cd "$FOUND_PROJECT"

export HYDRA_FULL_ERROR=1
export MNE_DATA="$EEG_DATA_ROOT"
CPU_TOTAL="${SLURM_CPUS_PER_TASK:-8}"
if ! [[ "${CPU_TOTAL}" =~ ^[0-9]+$ ]] || [[ "${CPU_TOTAL}" -le 0 ]]; then
  CPU_TOTAL=8
fi
CPU_PER_PROC=$((CPU_TOTAL / OUTER_PARALLEL))
if [[ "${CPU_PER_PROC}" -le 0 ]]; then
  CPU_PER_PROC=1
fi
export OMP_NUM_THREADS="${CPU_PER_PROC}"
export MKL_NUM_THREADS="${CPU_PER_PROC}"
export OPENBLAS_NUM_THREADS="${CPU_PER_PROC}"

RUN_BASE_DIR="${SLURM_SUBMIT_DIR:-${FOUND_PROJECT}}"
if [[ ! -d "${RUN_BASE_DIR}" ]]; then
  RUN_BASE_DIR="${FOUND_PROJECT}"
fi
if [[ ! -w "${RUN_BASE_DIR}" ]]; then
  RUN_BASE_DIR="${PWD}"
fi
ACC_FILE="${RUN_BASE_DIR}/acc.txt"
CHECKPOINT_DIR="${RUN_BASE_DIR}/.eeg_checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

if ! [[ "${MAX_RUNTIME_HOURS}" =~ ^[0-9]+$ ]] || [[ "${MAX_RUNTIME_HOURS}" -le 0 ]]; then
  echo "[FATAL] --max-runtime-hours must be a positive integer, got '${MAX_RUNTIME_HOURS}'" >&2
  exit 5
fi
if ! [[ "${OUTER_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${OUTER_PARALLEL}" -le 0 ]]; then
  echo "[FATAL] --outer-parallel must be a positive integer, got '${OUTER_PARALLEL}'" >&2
  exit 7
fi
if [[ "${FRAMEWORK}" != "tsmnet" && "${FRAMEWORK}" != "spdnet" ]]; then
  echo "[FATAL] --framework must be tsmnet|spdnet, got '${FRAMEWORK}'" >&2
  exit 6
fi

if [[ ! -f "conf/TSMNet/dataset/${DATASET_CFG}.yaml" ]]; then
  echo "[FATAL] dataset config not found: conf/TSMNet/dataset/${DATASET_CFG}.yaml" >&2
  exit 2
fi
if [[ ! -f "conf/TSMNet/evaluation/${EVALUATION_CFG}.yaml" ]]; then
  echo "[FATAL] evaluation config not found: conf/TSMNet/evaluation/${EVALUATION_CFG}.yaml" >&2
  exit 3
fi
if [[ ! -d "$EEG_DATA_ROOT" ]]; then
  echo "[FATAL] data root not found: ${EEG_DATA_ROOT}" >&2
  exit 4
fi
if [[ "${DATASET_CFG,,}" == "bnci2015_001" ]]; then
  BNCI2015_SOURCE_DIR="${MOABB_BNCI2015_001_DIR}"
  BNCI2015_EXPECTED_DIR="${EEG_DATA_ROOT}/MNE-bnci-data/~bci/database/001-2015"
  if [[ -d "${BNCI2015_SOURCE_DIR}" ]]; then
    mkdir -p "${BNCI2015_EXPECTED_DIR}"
    while IFS= read -r -d '' src_file; do
      ln -sfn "${src_file}" "${BNCI2015_EXPECTED_DIR}/$(basename "${src_file}")"
    done < <(find "${BNCI2015_SOURCE_DIR}" -maxdepth 1 -type f -name '*.mat' -print0)
  fi
fi

RUN_START_TS="$(date +%s)"
TIMEOUT_SECONDS=$((MAX_RUNTIME_HOURS * 3600))
TIMED_OUT=0

normalize_for_path() {
  echo "$1" | tr '/:+, ' '_____'
}

append_report() {
  local report_metric="$1"
  local raw_metric="$2"
  local seed="$3"
  local run_log="$4"
  local result_lines="$5"
  local result_paths="$6"
  local run_status="$7"
  local ts job_id results_one_line paths_one_line
  ts="$(date '+%F %T')"
  job_id="${SLURM_JOB_ID:-local}"
  results_one_line="$(printf "%s" "${result_lines}" | tr '\n' ' ' | tr '\t' ' ' | sed -E 's/[[:space:]]+/ /g')"
  paths_one_line="$(printf "%s" "${result_paths}" | tr '\n' ' ' | tr '\t' ' ' | sed -E 's/[[:space:]]+/ /g')"
  if [[ ! -s "${ACC_FILE}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "ts" "job_id" "dataset" "framework" "evaluation" "metric" "raw_metric" "distance_input" "seed" "use_lp" "use_logm" "swd_power" "lr" "loss_lr" "status" "result_path" "final_results" >> "${ACC_FILE}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${ts}" "${job_id}" "${DATASET_CFG}" "${FRAMEWORK}" "${EVALUATION_CFG}" "${report_metric}" "${raw_metric}" "${METRICS}" "${seed}" \
    "${USE_LP}" "${USE_LOGM}" "${SWD_POWER}" "${LR}" "${LOSS_LR}" "${run_status}" "${paths_one_line}" "${results_one_line}" >> "${ACC_FILE}"
}

IFS=',' read -r -a METRIC_LIST <<< "$METRICS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEED_LIST"
ACTIVE_PIDS=()
ACTIVE_STATUS_FILES=()
STOP_LAUNCH=0
FAILED_EXIT_CODE=0

run_single_task() {
  local metric="$1"
  local report_metric="$2"
  local seed="$3"
  local done_file="$4"
  local status_file="$5"
  local now_ts elapsed remaining run_log run_pid watchdog_pid run_exit_code
  local result_lines result_paths run_status timed_out_flag
  now_ts="$(date +%s)"
  elapsed=$((now_ts - RUN_START_TS))
  run_log="${RUN_BASE_DIR}/run_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)_${metric}_seed${seed}.log"
  if [[ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]]; then
    run_status="timeout"
    run_exit_code=0
    timed_out_flag=1
    result_lines="NO_FINAL_RESULTS_FOUND"
    result_paths="NO_RESULT_PATH_FOUND"
    append_report "${report_metric}" "${metric}" "${seed}" "${run_log}" "${result_lines}" "${result_paths}" "${run_status}"
    {
      echo "RUN_STATUS=${run_status}"
      echo "RUN_EXIT_CODE=${run_exit_code}"
      echo "TIMED_OUT_FLAG=${timed_out_flag}"
    } > "${status_file}"
    return 0
  fi
  remaining=$((TIMEOUT_SECONDS - elapsed))
  if [[ "${remaining}" -le 0 ]]; then
    run_status="timeout"
    run_exit_code=0
    timed_out_flag=1
    result_lines="NO_FINAL_RESULTS_FOUND"
    result_paths="NO_RESULT_PATH_FOUND"
    append_report "${report_metric}" "${metric}" "${seed}" "${run_log}" "${result_lines}" "${result_paths}" "${run_status}"
    {
      echo "RUN_STATUS=${run_status}"
      echo "RUN_EXIT_CODE=${run_exit_code}"
      echo "TIMED_OUT_FLAG=${timed_out_flag}"
    } > "${status_file}"
    return 0
  fi
  set +e
  if [[ "${FRAMEWORK}" == "tsmnet" ]]; then
    python TSMNet-MLR.py \
      hydra/launcher=joblib \
      hydra.launcher.n_jobs="${N_JOBS}" hydra.sweeper.max_batch_size="${MAX_BATCH_SIZE}" \
      dataset="${DATASET_CFG}" evaluation="${EVALUATION_CFG}" fit.data_dir="${EEG_DATA_ROOT}" fit.seed="${seed}" fit.device=GPU \
      nnet.model.swd_metric="${metric}" nnet.model.use_lp="${USE_LP}" nnet.model.use_logm="${USE_LOGM}" \
      nnet.model.n_proj="${N_PROJ}" nnet.model.swd_power="${SWD_POWER}" \
      nnet.model.loss_lambda1_init="${LAMBDA1}" nnet.model.loss_lambda2_init="${LAMBDA2}" \
      nnet.optimizer.lr="${LR}" nnet.optimizer.loss_lr="${LOSS_LR}" saving_model.is_save=True 2>&1 | tee "${run_log}" &
  else
    local spdnet_classifier="SPDMLR"
    local spdnet_metric="SPDLogEuclideanMetric"
    local spdnet_split_mode="random"
    if [[ "${EVALUATION_CFG}" == *"inter-session"* ]]; then
      spdnet_split_mode="session"
    elif [[ "${EVALUATION_CFG}" == *"inter-subject"* ]]; then
      spdnet_split_mode="subject"
    fi
    if [[ "${metric}" == "lsm" || "${metric}" == "olm" || "${metric}" == "spdsw" ]]; then
      spdnet_metric="SPDLogCholeskyMetric"
    fi
    if [[ "${metric}" == "logeig" || "${metric}" == "logeigmlr" ]]; then
      spdnet_classifier="LogEigMLR"
    fi
    python SPDNet-MLR.py \
      hydra/launcher=joblib \
      hydra.launcher.n_jobs=1 \
      dataset=RADAR dataset.name="${DATASET_CFG}" dataset.path="${EEG_DATA_ROOT}" fit.seed="${seed}" \
      +dataset.split_mode="${spdnet_split_mode}" \
      nnet.model.classifier="${spdnet_classifier}" nnet.model.metric="${spdnet_metric}" \
      +nnet.model.swd_metric="${metric}" +nnet.model.use_lp="${USE_LP}" +nnet.model.use_logm="${USE_LOGM}" \
      +nnet.model.n_proj="${N_PROJ}" +nnet.model.swd_power="${SWD_POWER}" \
      +nnet.model.loss_lambda1_init="${LAMBDA1}" +nnet.model.loss_lambda2_init="${LAMBDA2}" \
      +nnet.optimizer.loss_lr="${LOSS_LR}" \
      nnet.optimizer.lr="${LR}" fit.is_save=True 2>&1 | tee "${run_log}" &
  fi
  run_pid=$!
  (
    sleep "${remaining}"
    if kill -0 "${run_pid}" 2>/dev/null; then
      echo "__EEG_TIMEOUT__" >> "${run_log}"
      kill -TERM "${run_pid}" 2>/dev/null || true
      sleep 20
      kill -KILL "${run_pid}" 2>/dev/null || true
    fi
  ) &
  watchdog_pid=$!
  wait "${run_pid}"
  run_exit_code=$?
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
  set -e
  result_lines="$(grep -Ei 'final results:' "${run_log}" || true)"
  result_paths="$(grep -Ei 'results file path:' "${run_log}" || true)"
  if [[ -z "${result_lines}" ]]; then
    result_lines="NO_FINAL_RESULTS_FOUND"
  fi
  if [[ -z "${result_paths}" ]]; then
    result_paths="NO_RESULT_PATH_FOUND"
  fi
  run_status="ok"
  timed_out_flag=0
  if grep -q "__EEG_TIMEOUT__" "${run_log}"; then
    run_status="timeout"
    timed_out_flag=1
  elif [[ "${run_exit_code}" -ne 0 ]]; then
    run_status="failed_exit_${run_exit_code}"
  fi
  append_report "${report_metric}" "${metric}" "${seed}" "${run_log}" "${result_lines}" "${result_paths}" "${run_status}"
  if [[ "${run_status}" == "ok" ]]; then
    touch "${done_file}"
  fi
  {
    echo "RUN_STATUS=${run_status}"
    echo "RUN_EXIT_CODE=${run_exit_code}"
    echo "TIMED_OUT_FLAG=${timed_out_flag}"
  } > "${status_file}"
}

collect_finished_tasks() {
  local next_pids=()
  local next_status_files=()
  local i pid status_file task_exit task_timeout
  for i in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$i]}"
    status_file="${ACTIVE_STATUS_FILES[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_status_files+=("${status_file}")
      continue
    fi
    wait "${pid}" || true
    task_exit=1
    task_timeout=0
    if [[ -f "${status_file}" ]]; then
      task_exit="$(grep -E '^RUN_EXIT_CODE=' "${status_file}" | head -n1 | cut -d= -f2-)"
      task_timeout="$(grep -E '^TIMED_OUT_FLAG=' "${status_file}" | head -n1 | cut -d= -f2-)"
      rm -f "${status_file}"
    fi
    if [[ "${task_timeout}" == "1" ]]; then
      TIMED_OUT=1
      STOP_LAUNCH=1
    fi
    if [[ "${task_exit}" -ne 0 ]]; then
      FAILED_EXIT_CODE="${task_exit}"
      STOP_LAUNCH=1
    fi
  done
  ACTIVE_PIDS=("${next_pids[@]}")
  ACTIVE_STATUS_FILES=("${next_status_files[@]}")
}

for metric in "${METRIC_LIST[@]}"; do
  metric="$(echo "$metric" | tr -d '[:space:]')"
  if [[ -z "$metric" ]]; then
    continue
  fi
  REPORT_METRIC="${metric}"
  if [[ "${USE_LP,,}" == "false" && "${USE_LOGM,,}" == "true" ]]; then
    REPORT_METRIC="spdsw"
  fi
  for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "$seed" | tr -d '[:space:]')"
    if [[ -z "$seed" ]]; then
      continue
    fi
    safe_dataset="$(normalize_for_path "${DATASET_CFG}")"
    safe_eval="$(normalize_for_path "${EVALUATION_CFG}")"
    safe_metric="$(normalize_for_path "${REPORT_METRIC}")"
    safe_framework="$(normalize_for_path "${FRAMEWORK}")"
    safe_lp="$(normalize_for_path "${USE_LP}")"
    safe_logm="$(normalize_for_path "${USE_LOGM}")"
    safe_swd_power="$(normalize_for_path "${SWD_POWER}")"
    DONE_FILE="${CHECKPOINT_DIR}/done__${safe_framework}__${safe_dataset}__${safe_eval}__${safe_metric}__lp${safe_lp}__logm${safe_logm}__p${safe_swd_power}__seed${seed}.ok"
    if [[ "${RESUME}" -eq 1 && -f "${DONE_FILE}" ]]; then
      continue
    fi
    if [[ "${STOP_LAUNCH}" -eq 1 ]]; then
      break
    fi
    while [[ "${#ACTIVE_PIDS[@]}" -ge "${OUTER_PARALLEL}" ]]; do
      collect_finished_tasks
      sleep 2
    done
    STATUS_FILE="${RUN_BASE_DIR}/.eeg_task_status_${SLURM_JOB_ID:-local}_$(date +%s%N)_${metric}_seed${seed}.env"
    run_single_task "${metric}" "${REPORT_METRIC}" "${seed}" "${DONE_FILE}" "${STATUS_FILE}" &
    ACTIVE_PIDS+=("$!")
    ACTIVE_STATUS_FILES+=("${STATUS_FILE}")
  done
  if [[ "${STOP_LAUNCH}" -eq 1 ]]; then
    break
  fi
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  collect_finished_tasks
  sleep 2
done

if [[ "${FAILED_EXIT_CODE}" -ne 0 ]]; then
  exit "${FAILED_EXIT_CODE}"
fi

if [[ "${TIMED_OUT}" -eq 1 ]]; then
  if [[ "${AUTO_RESUBMIT}" -eq 1 ]]; then
    RESUBMIT_SCRIPT="${RUN_BASE_DIR}/eeg.sh"
    if [[ ! -f "${RESUBMIT_SCRIPT}" ]]; then
      RESUBMIT_SCRIPT="${FOUND_PROJECT}/eeg.sh"
    fi
    RESUBMIT_RAW="$(sbatch --parsable "${RESUBMIT_SCRIPT}" "${ORIGINAL_ARGS[@]}")"
    RESUBMIT_JOB_ID="${RESUBMIT_RAW%%;*}"
    echo "[AUTO-RESUBMIT] previous_job_id=${SLURM_JOB_ID:-local} new_job_id=${RESUBMIT_JOB_ID} script=${RESUBMIT_SCRIPT}"
    if [[ ! -s "${ACC_FILE}" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "ts" "job_id" "dataset" "framework" "evaluation" "metric" "raw_metric" "distance_input" "seed" "use_lp" "use_logm" "swd_power" "lr" "loss_lr" "status" "result_path" "final_results" >> "${ACC_FILE}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%F %T')" "${SLURM_JOB_ID:-local}" "${DATASET_CFG}" "${FRAMEWORK}" "${EVALUATION_CFG}" "-" "-" "-" "-" \
      "${USE_LP}" "${USE_LOGM}" "${SWD_POWER}" "${LR}" "${LOSS_LR}" "auto_resubmit:${RESUBMIT_JOB_ID}" "-" "-" >> "${ACC_FILE}"
    exit 0
  fi
  exit 0
fi

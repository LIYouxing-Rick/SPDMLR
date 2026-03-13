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

if [[ "${METRICS,,}" == "all" ]]; then
  METRICS="ecm,lecm,olm,lsm"
fi

if [[ "${FORCE_SPDSW}" -eq 1 ]]; then
  USE_LP="false"
  USE_LOGM="true"
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
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACC_FILE="${SCRIPT_DIR}/acc.txt"
CHECKPOINT_DIR="${SCRIPT_DIR}/.eeg_checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

if ! [[ "${MAX_RUNTIME_HOURS}" =~ ^[0-9]+$ ]] || [[ "${MAX_RUNTIME_HOURS}" -le 0 ]]; then
  echo "[FATAL] --max-runtime-hours must be a positive integer, got '${MAX_RUNTIME_HOURS}'" >&2
  exit 5
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
  {
    echo "===== $(date '+%F %T') ====="
    echo "job_id=${SLURM_JOB_ID:-local}"
    echo "dataset=${DATASET_CFG}"
    echo "framework=${FRAMEWORK}"
    echo "evaluation=${EVALUATION_CFG}"
    echo "metric=${report_metric}"
    echo "raw_metric=${raw_metric}"
    echo "distance_input=${METRICS}"
    echo "seed=${seed}"
    echo "seed_list=${SEED_LIST}"
    echo "n_jobs=${N_JOBS}"
    echo "max_batch_size=${MAX_BATCH_SIZE}"
    echo "use_lp=${USE_LP}"
    echo "use_logm=${USE_LOGM}"
    echo "n_proj=${N_PROJ}"
    echo "swd_power=${SWD_POWER}"
    echo "loss_lambda1_init=${LAMBDA1}"
    echo "loss_lambda2_init=${LAMBDA2}"
    echo "lr=${LR}"
    echo "loss_lr=${LOSS_LR}"
    echo "data_root=${EEG_DATA_ROOT}"
    echo "run_log=${run_log}"
    echo "run_status=${run_status}"
    echo "result_paths<<EOF"
    echo "${result_paths}"
    echo "EOF"
    echo "results<<EOF"
    echo "${result_lines}"
    echo "EOF"
    echo
  } >> "${ACC_FILE}"
}

IFS=',' read -r -a METRIC_LIST <<< "$METRICS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEED_LIST"
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
    DONE_FILE="${CHECKPOINT_DIR}/done__${safe_framework}__${safe_dataset}__${safe_eval}__${safe_metric}__seed${seed}.ok"

    if [[ "${RESUME}" -eq 1 && -f "${DONE_FILE}" ]]; then
      continue
    fi

    now_ts="$(date +%s)"
    elapsed=$((now_ts - RUN_START_TS))
    if [[ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]]; then
      TIMED_OUT=1
      break
    fi

    RUN_LOG="${SCRIPT_DIR}/run_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)_${metric}_seed${seed}.log"
    remaining=$((TIMEOUT_SECONDS - elapsed))
    if [[ "${remaining}" -le 0 ]]; then
      TIMED_OUT=1
      break
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
        nnet.optimizer.lr="${LR}" nnet.optimizer.loss_lr="${LOSS_LR}" saving_model.is_save=True 2>&1 | tee "${RUN_LOG}" &
    else
      SPDNET_CLASSIFIER="SPDMLR"
      SPDNET_METRIC="SPDLogEuclideanMetric"
      if [[ "${metric}" == "lsm" || "${metric}" == "olm" || "${metric}" == "spdsw" ]]; then
        SPDNET_METRIC="SPDLogCholeskyMetric"
      fi
      if [[ "${metric}" == "logeig" || "${metric}" == "logeigmlr" ]]; then
        SPDNET_CLASSIFIER="LogEigMLR"
      fi
      python SPDNet-MLR.py \
        hydra/launcher=joblib \
        hydra.launcher.n_jobs=1 \
        dataset=RADAR dataset.name="${DATASET_CFG}" dataset.path="${EEG_DATA_ROOT}" fit.seed="${seed}" \
        nnet.model.classifier="${SPDNET_CLASSIFIER}" nnet.model.metric="${SPDNET_METRIC}" \
        nnet.optimizer.lr="${LR}" fit.is_save=True 2>&1 | tee "${RUN_LOG}" &
    fi
    RUN_PID=$!
    (
      sleep "${remaining}"
      if kill -0 "${RUN_PID}" 2>/dev/null; then
        echo "__EEG_TIMEOUT__" >> "${RUN_LOG}"
        kill -TERM "${RUN_PID}" 2>/dev/null || true
        sleep 20
        kill -KILL "${RUN_PID}" 2>/dev/null || true
      fi
    ) &
    WATCHDOG_PID=$!
    wait "${RUN_PID}"
    RUN_EXIT_CODE=$?
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
    set -e

    RESULT_LINES="$(grep -Ei 'final results:' "${RUN_LOG}" || true)"
    RESULT_PATHS="$(grep -Ei 'results file path:' "${RUN_LOG}" || true)"
    if [[ -z "${RESULT_LINES}" ]]; then
      RESULT_LINES="NO_FINAL_RESULTS_FOUND"
    fi
    if [[ -z "${RESULT_PATHS}" ]]; then
      RESULT_PATHS="NO_RESULT_PATH_FOUND"
    fi
    RUN_STATUS="ok"
    if grep -q "__EEG_TIMEOUT__" "${RUN_LOG}"; then
      RUN_STATUS="timeout"
    elif [[ "${RUN_EXIT_CODE}" -ne 0 ]]; then
      RUN_STATUS="failed_exit_${RUN_EXIT_CODE}"
    fi

    append_report "${REPORT_METRIC}" "${metric}" "${seed}" "${RUN_LOG}" "${RESULT_LINES}" "${RESULT_PATHS}" "${RUN_STATUS}"
    if [[ "${RUN_STATUS}" == "timeout" ]]; then
      TIMED_OUT=1
      break
    fi
    if [[ "${RUN_EXIT_CODE}" -ne 0 ]]; then
      exit "${RUN_EXIT_CODE}"
    fi
    touch "${DONE_FILE}"
  done
  if [[ "${TIMED_OUT}" -eq 1 ]]; then
    break
  fi
done

if [[ "${TIMED_OUT}" -eq 1 ]]; then
  if [[ "${AUTO_RESUBMIT}" -eq 1 ]]; then
    sbatch "${SCRIPT_DIR}/eeg.sh" "${ORIGINAL_ARGS[@]}"
    exit 0
  fi
  exit 0
fi

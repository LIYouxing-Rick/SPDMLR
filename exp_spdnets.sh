#!/bin/bash
set -euo pipefail

TARGET_DATASET="${1:-ALL}"
RADAR_PATH="${2:-/data}"
HDM05_PATH="${3:-/data}"

if [[ "$TARGET_DATASET" == "RADAR" || "$TARGET_DATASET" == "ALL" ]]; then
  python SPDNet-MLR.py -m dataset=RADAR dataset.path="$RADAR_PATH" nnet.model.architecture=[20,16,14,12,10,8],[20,16,8] nnet.model.classifier=LogEigMLR
  python SPDNet-MLR.py -m dataset=RADAR dataset.path="$RADAR_PATH" nnet.model.architecture=[20,16,14,12,10,8],[20,16,8] nnet.model.classifier=SPDMLR nnet.model.metric=SPDLogEuclideanMetric nnet.model.beta=1.,0.
  python SPDNet-MLR.py -m dataset=RADAR dataset.path="$RADAR_PATH" nnet.model.architecture=[20,16,14,12,10,8],[20,16,8] nnet.model.classifier=SPDMLR nnet.model.metric=SPDLogCholeskyMetric nnet.model.power=1.,0.5
fi

if [[ "$TARGET_DATASET" == "HDM05" || "$TARGET_DATASET" == "ALL" ]]; then
  python SPDNet-MLR.py -m dataset=HDM05 dataset.path="$HDM05_PATH" nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=LogEigMLR
  python SPDNet-MLR.py -m dataset=HDM05 dataset.path="$HDM05_PATH" nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=SPDMLR nnet.model.metric=SPDLogEuclideanMetric
  python SPDNet-MLR.py -m dataset=HDM05 dataset.path="$HDM05_PATH" nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=SPDMLR nnet.model.metric=SPDLogCholeskyMetric nnet.model.power=1.,0.5
fi

if [[ "$TARGET_DATASET" != "RADAR" && "$TARGET_DATASET" != "HDM05" && "$TARGET_DATASET" != "ALL" ]]; then
  echo "Invalid dataset: $TARGET_DATASET"
  echo "Usage: bash exp_spdnets.sh [RADAR|HDM05|ALL] [RADAR_PATH] [HDM05_PATH]"
  exit 1
fi

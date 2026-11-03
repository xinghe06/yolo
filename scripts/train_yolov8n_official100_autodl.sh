#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ULTRA_DIR="${ULTRA_DIR:-${PROJECT_ROOT}/third_party/ultralytics}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/VisDrone}"
GPU_ID="${GPU_ID:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-N10_yolov8n_official100}"
RUN_PROJECT="${RUN_PROJECT:-${PROJECT_ROOT}/runs/paper}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
DATA_YAML="${PROJECT_ROOT}/configs/VisDrone-autodl.yaml"

TRAIN_IMAGES="${DATA_ROOT}/VisDrone2019-DET-train/images"
TRAIN_LABELS="${DATA_ROOT}/VisDrone2019-DET-train/labels"
VAL_IMAGES="${DATA_ROOT}/VisDrone2019-DET-val/images"
VAL_LABELS="${DATA_ROOT}/VisDrone2019-DET-val/labels"

for required_path in \
  "${ULTRA_DIR}/ultralytics/cfg/default.yaml" \
  "${ULTRA_DIR}/yolov8n.pt" \
  "${TRAIN_IMAGES}" \
  "${TRAIN_LABELS}" \
  "${VAL_IMAGES}" \
  "${VAL_LABELS}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

if [[ -n "$(git -C "${ULTRA_DIR}" status --porcelain)" ]]; then
  echo "The baseline Ultralytics checkout is not clean: ${ULTRA_DIR}" >&2
  git -C "${ULTRA_DIR}" status --short >&2
  exit 1
fi

ULTRA_VERSION="$(git -C "${ULTRA_DIR}" describe --tags --always)"
if [[ "${ULTRA_VERSION}" != "v8.0.239" ]]; then
  echo "Expected Ultralytics v8.0.239, found ${ULTRA_VERSION}" >&2
  exit 1
fi

TRAIN_IMAGE_COUNT="$(find "${TRAIN_IMAGES}" -type f -name '*.jpg' | wc -l)"
TRAIN_LABEL_COUNT="$(find "${TRAIN_LABELS}" -type f -name '*.txt' | wc -l)"
VAL_IMAGE_COUNT="$(find "${VAL_IMAGES}" -type f -name '*.jpg' | wc -l)"
VAL_LABEL_COUNT="$(find "${VAL_LABELS}" -type f -name '*.txt' | wc -l)"

if [[ "${TRAIN_IMAGE_COUNT}" != "6471" || "${TRAIN_LABEL_COUNT}" != "6471" || \
      "${VAL_IMAGE_COUNT}" != "548" || "${VAL_LABEL_COUNT}" != "548" ]]; then
  echo "Unexpected VisDrone file counts:" >&2
  echo "  train images=${TRAIN_IMAGE_COUNT}, labels=${TRAIN_LABEL_COUNT}" >&2
  echo "  val   images=${VAL_IMAGE_COUNT}, labels=${VAL_LABEL_COUNT}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/configs" "${LOG_DIR}" "${RUN_PROJECT}"
mkdir -p "${PROJECT_ROOT}/.cache/matplotlib" "${PROJECT_ROOT}/.cache/ultralytics"

cat >"${DATA_YAML}" <<EOF
path: "${DATA_ROOT}"
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
EOF

export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${PROJECT_ROOT}/.cache/ultralytics"
export PYTHONPATH="${ULTRA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ULTRA_DIR}"

python - <<'PY'
import sys
import torch
import ultralytics

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"Ultralytics source: {ultralytics.__file__}")
assert ultralytics.__version__ == "8.0.239"
assert torch.cuda.is_available(), "CUDA is not available"
PY

echo "Dataset: train=${TRAIN_IMAGE_COUNT}, val=${VAL_IMAGE_COUNT}"
echo "GPU_ID=${GPU_ID}, workers=${WORKERS}"
echo "Output: ${RUN_PROJECT}/${RUN_NAME}"

yolo detect train \
  model="${ULTRA_DIR}/yolov8n.pt" \
  data="${DATA_YAML}" \
  epochs=100 \
  patience=50 \
  batch=16 \
  imgsz=640 \
  optimizer=auto \
  workers="${WORKERS}" \
  seed=0 \
  deterministic=True \
  pretrained=True \
  close_mosaic=10 \
  device="${GPU_ID}" \
  project="${RUN_PROJECT}" \
  name="${RUN_NAME}" \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

#!/usr/bin/env bash
# Creates .venv, installs pinned deps, fetches the Panda model and the MediaPipe hand model.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
MENAGERIE_COMMIT=62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt

if [ ! -f third_party/mujoco_menagerie/franka_emika_panda/panda.xml ]; then
  mkdir -p third_party
  rm -rf third_party/mujoco_menagerie
  git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
  git -C third_party/mujoco_menagerie checkout -q "$MENAGERIE_COMMIT"
fi

mkdir -p models
if [ ! -f models/hand_landmarker.task ]; then
  curl -sSL -o models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi
echo "setup done"

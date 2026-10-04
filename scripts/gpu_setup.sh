#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run inside the SGLang v0.5.21 CUDA 13 Linux environment (see GPU_RUNBOOK).
python3 -c 'import sys; assert sys.version_info[:2] == (3,12)'
nvidia-smi
bash scripts/bootstrap.sh
python3 -m pip install --only-binary=:all: --require-hashes -r configs/gpu-requirements.lock
bash scripts/build_native.sh
python3 -m pip install --no-deps --no-build-isolation -e ./sglang/python
python3 -m pip check
mkdir -p results/environment
python3 -m pip freeze > results/environment/pip-freeze.txt
nvidia-smi -q > results/environment/nvidia-smi.txt
python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

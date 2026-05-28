#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vrcd}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu124}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create --name "${ENV_NAME}" python=3.11 pip -y \
    -c https://repo.anaconda.com/pkgs/main \
    -c https://repo.anaconda.com/pkgs/r
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install --extra-index-url "${CUDA_WHEEL_INDEX}" \
  torch==2.6.0 \
  torchvision==0.21.0

conda run -n "${ENV_NAME}" python -m pip install -e '.[inference]'

conda run -n "${ENV_NAME}" python -c \
  "import torch, transformers; print('python/torch environment is ready'); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('transformers:', transformers.__version__)"

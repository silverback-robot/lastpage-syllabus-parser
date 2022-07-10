#!/bin/bash

# Check for sudo privileges
if [ "$EUID" -ne 0 ]
  then echo "Please run as root/sudo"
  exit 2
fi

# Check if running inside Python virtual env
STR=`pip -V`
SUB=`pwd`

if [[ "$STR" != *"$SUB"* ]]; then
  echo "Activating virtual env (venv)"
  source ./venv/bin/activate
  echo `pip -V`
fi

# Poppler (PDF handling)
apt install -y poppler-utils

## Ninja build system for faster builds of Detectron2 (optional)
apt-get install ninja-build

## PyTorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

## OpenCV for visualization (optional)
pip install "opencv-python-headless<4.3" # https://github.com/opencv/opencv-python/issues/591

# Install Detectron2 from source
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install known python requirements
pip install -r requirements.txt

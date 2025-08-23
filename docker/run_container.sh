#!/usr/bin/env bash
set -e
docker run --gpus all --rm -it \
  -v "$(cd .. && pwd)":/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.06-py3 bash

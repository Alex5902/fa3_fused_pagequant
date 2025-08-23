FROM nvcr.io/nvidia/pytorch:24.06-py3

# Basic build deps
RUN apt-get update && apt-get install -y \
    git ninja-build build-essential ccache tmux pciutils jq \
    libaio-dev && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/root/.ccache CC="ccache gcc" CXX="ccache g++"

RUN pip install --upgrade pip wheel setuptools
RUN pip install triton==2.3.0 packaging
RUN pip install numpy scipy pandas matplotlib
RUN pip install nvidia-ml-py pynvml

# Optional: vLLM
RUN pip install vllm==0.5.4 transformers==4.43.3 accelerate

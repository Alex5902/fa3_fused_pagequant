# FA3-FPQ: FlashAttention-3 with Fused PageQuant

[![GPU - CUDA 12.8+](https://img.shields.io/badge/GPU-CUDA%2012.8%2B-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture - Hopper](https://img.shields.io/badge/Architecture-Hopper-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/en-us/data-center/hopper-architecture/)
[![Language - C++/Python](https://img.shields.io/badge/Language-C%2B%2B%20%26%20Python-blue?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

This repository contains a high-performance, research-focused implementation of **FlashAttention-3 (FA3)**, specifically optimized for **NVIDIA Hopper-architecture GPUs (H100/H200, sm_90a)**. The core feature is the integration of **Fused PageQuant (FPQ)**, a custom CUDA kernel that performs on-the-fly dequantization of a paged INT8 Key-Value cache directly into shared memory, minimizing latency and memory bandwidth.

This project was developed and validated on the ABCI supercomputing infrastructure, and includes PBS Pro job scripts, custom testing harnesses, and Python wheel packaging for streamlined deployment.

---

## 🎯 Current Status & Next Steps

The project is in the final stages of kernel integration. The core FPQ CUDA logic for dequantizing INT8 K/V tensors is implemented, and the necessary modifications to the FlashAttention C++ API have been made.

-   **Current Blocker:** The build is failing due to a subtle C++ template type mismatch when passing `cute::Layout` objects for the quantization scales and zero-points from the host-side launch code to the device-side kernel code. This is a common and challenging issue when plumbing complex template types through the FA3/CUTLASS architecture.
-   **Immediate Next Step:** Resolve the `no suitable user-defined conversion` error in `flash_fwd_launch_template.h`. This involves ensuring the `cute::Layout` type definition in `mainloop_fwd_sm90_tma_gmma_ws.hpp` exactly matches the type being constructed and passed during kernel launch.
-   **Post-Build Plan:**
    1.  **Correctness:** Run the full correctness suite (`fa3_correctness.py`) to validate the fused dequantization logic against both a naive PyTorch implementation and the standard FP16 FA3 implementation.
    2.  **Performance Profiling:** Use the micro-benchmarking harness (`microbench.py`) with `nsys` to profile the new kernel, ensuring the dequantization overhead is minimal and that TMA/GMMA pipelines remain efficient.
    3.  **End-to-End Benchmarking:** Evaluate performance on long-context inference scenarios and within a full VLLM integration to measure real-world speedups.

---

## 📂 Repository Layout

-   **`flash-attention/`**: The upstream FlashAttention-3 codebase, patched to support the custom kernel build and FPQ data types. The core development happens within `flash-attention/hopper/`.
-   **`kernels/`**: Standalone prototype CUDA kernels for the fused dequantization logic (`fa3_fused_dequant.cuh`).
-   **`harness/`**: Python scripts for comprehensive testing, including unit tests (`pytest`), correctness validation against reference implementations, performance micro-benchmarks, and end-to-end profiling.
-   **`kv_formats/`**: Utilities for creating and managing paged KV cache formats, including page quantization tools for generating FP8/INT8 test data.
-   **`pbs/`**: Job submission scripts for the PBS Pro workload manager, easily adaptable for Slurm.
-   **`patches/`**: Patch files for integrating FA3-FPQ into downstream libraries like VLLM.
-   **`logs/`**: Build and runtime logs for debugging and reproducibility (generally not committed).
-   **`wheels/`**: Pre-built Python `.whl` packages for fast, reproducible installation in target environments.
-   **`infra/`**: Project infrastructure, including `.gitconfig` for the repository and utility scripts.

---

## 🚀 Core Features

-   **Hopper-Native Architecture**: Built exclusively for `sm_90a`, leveraging Tensor Memory Accelerator (TMA) for asynchronous data movement and Grace-Hopper's GMMA units for matrix operations.
-   **Fused PageQuant Kernel**: Custom INT8 dequantization logic is fused directly into the main FA3 forward kernel. K/V tensors are loaded from global memory (HBM) and dequantized on-the-fly into shared memory, avoiding intermediate FP16 tensors.
-   **Comprehensive Testing Harness**: Includes tools for correctness validation, performance micro-benchmarking, and end-to-end profiling.
-   **VLLM Integration**: Provides patches and scripts to integrate the custom FA3-FPQ build into a VLLM environment for realistic end-to-end performance evaluation.
-   **Reproducible Supercomputing Workflow**: Contains scripts and configurations for building and testing on HPC clusters using environments like PBS Pro or Slurm.

---

## ⚙️ Setup and Installation

### Prerequisites

-   **GPU:** NVIDIA Hopper H100 or H200 (`sm_90a`)
-   **CUDA Toolkit:** 12.8+
-   **PyTorch:** 2.3+ (built with CUDA 12.1+ support)
-   **Compiler:** g++ (compatible with your CUDA version)
-   **Build Tools:** `ninja`

### Option A: Install from a Pre-built Wheel

This is the recommended method for quick deployment in a matching environment.

```bash
# Activate your Python environment
pip install wheels/fa3_fpq-3.0.0-cp311-cp311-linux_x86_64.whl
```

### Option B: Build from Source

Use the provided build script, which handles environment setup and compilation.

```bash
# 1. Navigate to the build directory
cd flash-attention/hopper

# 2. Run the build script
# (This script should handle module loading and environment variables)
bash build-fa3.sh
```

---

## 🧪 Testing and Validation

The harness provides scripts to ensure the correctness and performance of the custom kernel.

### Correctness Checks

You can run correctness tests locally or submit them as a batch job.

```bash
# Navigate to the testing harness
cd harness/

# Run pytest for basic unit tests
python -m pytest

# Run the comprehensive correctness check against PyTorch reference
python fa3_correctness.py
```

To run on a cluster node:

```bash
# Submit the correctness job
qsub pbs/fa3_correctness.pbs
```

---

## 📊 Benchmarking and Profiling

The harness includes scripts for detailed performance analysis.

-   **Long-Context Inference:** `harness/eval_longcontext.py` measures performance on long sequences.
-   **VLLM End-to-End:** `harness/e2e_bench_vllm.py` evaluates the full system performance with the VLLM patch applied.
-   **Kernel Microbenchmarks:** `harness/microbench.py` allows for targeted profiling of the CUDA kernel using tools like `nsys`.

---

## 📌 Notes

-   Developed and tested on **ABCI** with PBS Pro, CUDA 12.8, and PyTorch 2.3.1.
-   The build is currently configured for a **minimal set of head dimensions** to accelerate compilation during development. This can be expanded in `setup.py`.

---

## 📜 License

The modifications and custom code in this repository are © 2025 Alejandro Ito Aramendia. The underlying FlashAttention and CUTLASS codebases are subject to their original licenses. See the `flash-attention/LICENSE` file for details.

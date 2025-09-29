# Raspberry Pi Setup
## Update and install packages
```bash
sudo apt update
sudo apt install -y git build-essential cmake pkg-config libopenblas-dev jq libcurl4-openssl-dev libssl-dev
```
- `libcurl4-openssl-dev` `libssl-dev` for curl headers and libs, required for compilation

## Setup Conda environment
```bash
curl -fsSLo ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash ~/miniforge.sh -b -p "$HOME/miniforge3"
rm ~/miniforge.sh
```
```bash
conda create -n llmbench python=3.11 -y
conda activate llmbench
pip install -U huggingface_hub
```

## Build llama.cpp for CPU
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -S . -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build -j"$(nproc)"
```
# Models
**Maximum model parameters:** 4 billion parameters (4b)
**Imatrix calibration file:** https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8, found from [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) (used for quantisations)
Note: uses base rather than instruct models where possible, though shouldn't affect performance anyway
**Quantisations to test:** 
- Q3_K_M
- Q4_K_S, Q4_K_M
- Q5_K_S, Q5_K_M
## Model List
- [Llama3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf): [1b](https://huggingface.co/meta-llama/Llama-3.2-1B), [3b](https://huggingface.co/meta-llama/Llama-3.2-3B)
- Note: [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) hosts Q4_0_8_8, Q4_0_4_8, Q4_0_4_4 quants which are apparently good for ARM chips
- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f): [0.6b](https://huggingface.co/Qwen/Qwen3-0.6B), [1.7b](https://huggingface.co/Qwen/Qwen3-1.7B), [4b](https://huggingface.co/Qwen/Qwen3-4B)
	- Pre-quantised (Q8_0): [0.6b](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), [1.7b](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF), [4b](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Gemma-3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d): [270m](https://huggingface.co/google/gemma-3-270m), [2b](https://huggingface.co/google/gemma-3-1b-pt), [4b](https://huggingface.co/google/gemma-3-4b-pt)
	- Unquantised, Quantisation Aware Trained (Q4_0): [270m](https://huggingface.co/google/gemma-3-270m-qat-q4_0-unquantized)
	- Pre-quantised, Quantisation Aware Trained (Q4_0): [1b](https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf), [4b](https://huggingface.co/google/gemma-3-4b-pt-qat-q4_0-gguf)
- BitNet b1.58 (NOTE: not implemented, requires different [BitNet inference framework](https://github.com/microsoft/BitNet) instead of llama.cpp)

## Installation
todo

## Quantisation
- [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo): converts public repo to a quantised GGUF version on a HuggingFace Space
# TODO:
- completely rework llm-bench shell script
- benchmark models from ./models/*, save each run to a csv
- update README with how to set-up llm-bench

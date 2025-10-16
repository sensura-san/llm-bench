# llm-bench
A general bash script for easily benchmarking multiple models based on prompt processing and generation speed. Built atop llama-bench. 
## Project focus 
Benchmarking different quantisations of LLMs for edge device computing (specifically, the Raspberry Pi 4b) for prompt processing / token generation speed.
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
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda init bash   # optional, makes it permanent for future shells
exec bash         # reload shell to pick up init (or start a new terminal)
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
**Suggested maximum model parameters:** 4 billion parameters (4b)

Note: use instruct models where possible

**Quantisations to test:** 
- Q3_K_M
- Q4_K_S, Q4_K_M
- Q5_K_S, Q5_K_M
## Model List
- Qwen2.5: 0.5b
- [Llama3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf): [1b](https://huggingface.co/meta-llama/Llama-3.2-1B), [3b](https://huggingface.co/meta-llama/Llama-3.2-3B)
- Note: [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) hosts Q4_0_8_8, Q4_0_4_8, Q4_0_4_4 quants which are apparently good for ARM chips
- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f): [0.6b](https://huggingface.co/Qwen/Qwen3-0.6B), [1.7b](https://huggingface.co/Qwen/Qwen3-1.7B), [4b](https://huggingface.co/Qwen/Qwen3-4B)
	- Pre-quantised (Q8_0): [0.6b](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), [1.7b](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF), [4b](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Gemma-3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d): [270m](https://huggingface.co/google/gemma-3-270m), [2b](https://huggingface.co/google/gemma-3-1b-pt), [4b](https://huggingface.co/google/gemma-3-4b-pt)
	- Unquantised, Quantisation Aware Trained (Q4_0): [270m](https://huggingface.co/google/gemma-3-270m-qat-q4_0-unquantized)
	- Pre-quantised, Quantisation Aware Trained (Q4_0): [1b](https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf), [4b](https://huggingface.co/google/gemma-3-4b-pt-qat-q4_0-gguf)
 - [Gemma-3n](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4): [e2b (6b)](https://huggingface.co/google/gemma-3n-E2B-it)
    - e2b (6b) means that while it has 6 bil. parameters but effectively loads only 2 bil. params. into memory by using a novel architecture, thus reducing memory footprint
    - Optimised for CPU/mobile devices, so possibly good for edge
- BitNet b1.58 (NOTE: not implemented, requires different [BitNet inference framework](https://github.com/microsoft/BitNet) instead of llama.cpp)

## Installation Example
```bash
 hf download Qwen/Qwen3-0.6B --local-dir models/qwen3-06b
```

## Quantisation
```bash
# from Huggingface, obtain the official meta-llama/Llama-3.1-8B model weights and place them in ./models
ls ./models
config.json             model-00001-of-00004.safetensors  model-00004-of-00004.safetensors  README.md                tokenizer.json
generation_config.json  model-00002-of-00004.safetensors  model.safetensors.index.json      special_tokens_map.json  USE_POLICY.md
LICENSE                 model-00003-of-00004.safetensors  original                          tokenizer_config.json

# [Optional] for PyTorch .bin models like Mistral-7B
ls ./models
<folder containing weights and tokenizer json>

# install Python dependencies
python3 -m pip install -r requirements.txt

# convert the model to ggml FP16 format
python3 convert_hf_to_gguf.py ./models/mymodel/

# quantize the model to 4-bits (using Q4_K_M method)
./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

# update the gguf filetype to current version if older version is now unsupported
./llama-quantize ./models/mymodel/ggml-model-Q4_K_M.gguf ./models/mymodel/ggml-model-Q4_K_M-v2.gguf COPY
```
### Or alternatively do it online:
- [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo): converts public repo to a quantised GGUF version on a HuggingFace Space

# Test a model
```bash
./llama-cli -m models/qwen3-06b/qwen3-752M-06b-Q4_K_M.gguf \
  -sys "You are a helpful assistant"
```
```bash
./llama-bench -m models/qwen3-06b/qwen3-752M-06b-Q4_K_M.gguf \
  -p 128 -n 64 -t 4 -r 1 -o json | jq .
```

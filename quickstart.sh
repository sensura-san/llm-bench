#!/usr/bin/env bash
cd /media/ic2/ESD-USB/llm-bench || exit
export LLAMA="/home/ic2/llm-bench/llama.cpp/build/bin/llama-bench"
export MODEL_DIR="/media/ic2/ESD-USB/models"
bash ./llm-bench

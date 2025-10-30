#!/bin/bash

# Start dots.ocr vLLM server
echo "🚀 Starting dots.ocr vLLM server (vLLM 0.11.0+ with official DotsOCR support)..."

# Set model path
export hf_model_path=./dots.ocr/weights/DotsOCR

# Check if model weights exist
if [ ! -d "$hf_model_path" ]; then
    echo "❌ DotsOCR model weights not found at $hf_model_path"
    echo "Please download the model weights first:"
    echo "cd dots.ocr && python3 tools/download_model.py"
    exit 1
fi

echo "✅ Model weights found at $hf_model_path"

# Launch vLLM server with official DotsOCR support (vLLM 0.11.0+)
# No need for out-of-tree registration - DotsOCR is built into vLLM 0.11.0+
echo "🔥 Launching vLLM server on port 8033 (using GPU 1)..."
CUDA_VISIBLE_DEVICES=1 vllm serve ${hf_model_path} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code \
    --port 8033
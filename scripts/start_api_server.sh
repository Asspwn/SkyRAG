#!/bin/bash

# Start FastAPI server
echo "🚀 Starting FastAPI RAG service..."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. LLM features (/ask endpoint) will be disabled."
    echo "To enable LLM features, set OPENAI_API_KEY in .env file or environment variable."
else
    echo "✅ OpenAI API key configured (using model: ${OPENAI_MODEL:-gpt-4o-mini})"
fi

# Check if vLLM server is running
echo "🔍 Checking if vLLM server is running on port 8033..."
if ! curl -s http://localhost:8033/health >/dev/null 2>&1; then
    echo "❌ vLLM server is not running on port 8033"
    echo "Please start the vLLM server first:"
    echo "./scripts/start_vllm_server.sh"
    echo ""
    echo "Continuing anyway (server will fail when processing documents)..."
fi

# Start the API server
echo "🔥 Starting FastAPI server on port 8034 (using GPU 1 for embeddings)..."
CUDA_VISIBLE_DEVICES=1 python src/api.py
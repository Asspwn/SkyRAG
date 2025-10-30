#!/bin/bash

# Environment setup script for dots.ocr + Haystack RAG platform
echo "🔧 Setting up dots.ocr + Haystack RAG environment..."

# # Check if we're in the correct conda environment
# if [[ "$CONDA_DEFAULT_ENV" != "haystack_ai_env" ]]; then
#     echo "❌ Please activate the haystack_ai_env conda environment first:"
#     echo "conda activate haystack_ai_env"
#     exit 1
# fi

echo "✅ Current environment: $CONDA_DEFAULT_ENV"

# Download dots.ocr model weights if they don't exist
DOTS_OCR_WEIGHTS="./dots.ocr/weights/DotsOCR"
if [ ! -d "$DOTS_OCR_WEIGHTS" ]; then
    echo "📥 Downloading dots.ocr model weights..."
    cd dots.ocr
    python3 tools/download_model.py
    cd ..
    echo "✅ Model weights downloaded"
else
    echo "✅ dots.ocr model weights already exist"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
echo "✅ Directories created"

# Test imports
echo "🧪 Testing Python imports..."
python -c "
try:
    import haystack
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    import sentence_transformers
    import fastapi
    import dots_ocr
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Start vLLM server: ./scripts/start_vllm_server.sh"
echo "2. Start API server: ./scripts/start_api_server.sh"
echo "3. Test the API: python tests/test_api.py"
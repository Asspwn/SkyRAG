# Multi-User RAG Platform with DotsOCR

A production-ready Retrieval-Augmented Generation (RAG) system with PDF document processing using DotsOCR vision-language model. Each user gets isolated document storage and vector embeddings for semantic search.

Link to the DEMO:
https://leonida-interrupted-gracelyn.ngrok-free.dev/
I have already indexed pdfs and they are stored in vector store. Please choose the User id: **skyro1** or **demo_user** and feel free to test it.

## Features

- **PDF Document Processing** - Upload and parse PDFs with DotsOCR (VLM-based OCR)
- **Semantic Search** - Find relevant information across documents
- **LLM-Powered Q&A** - Ask questions and get AI-generated answers from your documents
- **Multimodal RAG** - Vision-enabled LLM analyzes both text AND images from PDFs
- **Multi-User Support** - Isolated storage per user
- **Image Extraction** - Automatically extract and serve images from PDFs
- **FastAPI Backend** - Modern async API with auto-documentation

## Models & Frameworks

  - **DotsOCR (VLM)** - Vision-language model for advanced PDF parsing
  - **multilingual-e5-large-instruct** - Embedding model for semantic search
  - **GPT-4o-mini** - OpenAI's vision-enabled LLM for Q&A
  - **Haystack 2.18** - Modular NLP framework for RAG pipelines
  - **Qdrant** - Vector database for efficient similarity search
  - **vLLM v0.11.0** - High-performance inference server

## Other Models

The following chart (LongBench v2 results) shows how different LLMs perform on long-context reasoning tasks.
LongBench v2 is a benchmark introduced by Bai et al. (2024) (arXiv:2412.15204 v2) designed to evaluate deep understanding and reasoning across long-context multitasks such as:

  - single and multi-document QA
  - long in-context learning
  - dialogue history understanding
  - code-repository and structured-data comprehension

Models close to or above human baseline can be used as LLM for RAG given that it can handle Filipono and English language and have multimodal capabilities to describe images.
  
  <img width="1105" height="626" alt="image" src="https://github.com/user-attachments/assets/0e704e3a-3e64-4d4b-b11b-beab908d2daa" />
 

---
## Discussion of the solution

First of all I wanted to create a RAG that can be easily manageable with features like document upload, indexing and deletion.

I stick to the Haystack framework, as their new version has more clean coding pipeline components. 

In my opinion, one of the biggest concerns of RAG is clean data. There is a relatively new OCR model which is backed by a small VLM called DotsOCR. I used it to parse pdf files because it can easily convert tables to markdowns and extract images. Images are saved in the vector store of the user so that it can be interpreted by LLM if it extracted a relevant chunk of the document which has the image.

Skyro can extend it to scale solutions both for the internal documentation as well as for clients to get answers faster and more reliable.

Integration can happen on the level of API. For the demo purposes I have created a simple UI using the streamlit.

For the product team the solution can be presented as a DEMO and further features updates can be discussed. Based on that, the Backend and Frontend engineers and AI engineers can work on a basis of sprints to deploy this solution to the solid production level.

Both API and Open sourced LLMs can be used.
The requirements to the LLMs are:
- Multimodal 
- Multilingual (Filipino and English are primal languages)
- Enough context size (at least 4096 due to images)



## Architecture
<img width="682" height="242" alt="image" src="https://github.com/user-attachments/assets/f6bfc072-9dbb-48c2-90ae-110bd3329b60" />

<img width="644" height="342" alt="image" src="https://github.com/user-attachments/assets/54675e54-4da8-40bc-a530-4ddb325d2583" />

### Components

1. **vLLM Server** - Serves DotsOCR VLM model for document parsing
2. **RAG API** - FastAPI server handling:
   - Document upload & processing
   - Vector embedding (multilingual-e5-large-instruct)
   - Qdrant-based semantic search
   - Multi-user data management

---

## Quick Start

### Manual Setup

```bash
# 1. Setup environment
conda create -n rag_env python=3.11 -y
conda activate rag_env
pip install -r requirements.txt
pip install -e ./dots.ocr

# 2. Download model weights
cd dots.ocr && python3 tools/download_model.py && cd ..

# 3. Configure OpenAI API (for LLM features)
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your-key-here
# OPENAI_MODEL=gpt-4o-mini  # or gpt-4o for better vision

# 4. Start vLLM server (Terminal 1)
./scripts/start_vllm_server.sh

# 5. Start API (Terminal 2)
./scripts/start_api_server.sh
```

**📖 Full Manual Guide:** See `DEPLOYMENT.md`

---
## Usage Example

### Upload Documents

```bash
curl -X POST "http://localhost:8034/api/users/john/documents/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### Search Documents (Raw Chunks)

```bash
curl -X POST "http://localhost:8034/api/users/john/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "top_k": 5
  }'
```

### Ask Questions (LLM-Generated Answers) 🆕

```bash
# Text + Vision (analyzes images from PDFs)
curl -X POST "http://localhost:8034/api/users/john/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings in the charts?",
    "top_k": 5,
    "temperature": 0.7,
    "include_images": true
  }' | jq

# Text-only mode (faster, cheaper)
curl -X POST "http://localhost:8034/api/users/john/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key points",
    "top_k": 5,
    "include_images": false
  }' | jq
```

**Response:**
```json
{
  "answer": "Based on the documents, the main findings are... [Document 1]",
  "sources": [
    {
      "document_number": 1,
      "content": "...",
      "score": 0.85,
      "images": ["data/john/images/chart1.png"]
    }
  ],
  "question": "What are the main findings?",
  "model": "gpt-4o-mini",
  "retrieved_documents": 5,
  "images_analyzed": 3,
  "total_images_available": 3
}
```

### List Documents

```bash
curl http://localhost:8034/api/users/john/documents
```

### Delete Documents

```bash
curl -X DELETE "http://localhost:8034/api/users/john/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "filenames": ["document1.pdf", "document2.pdf"]
  }'
```

### Get User Stats

```bash
curl http://localhost:8034/api/users/john/stats
```

### Get Extracted Images

```bash
# List images
curl http://localhost:8034/api/users/john/images

# Get specific image
curl http://localhost:8034/api/users/john/images/doc1_page_1.jpg -o image.jpg
```


## Requirements

### Hardware

- **GPU:** 16GB+ VRAM (RTX 4080, A4000, or better)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 30GB+ free space

---

## Project Structure

```
SkyRAG/
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── DEPLOYMENT.md                       # Deployment guide
├── .env.example                        # Environment variables template
│
├── src/                                # Core application code
│   ├── api.py                         # FastAPI server with endpoints
│   ├── rag_service.py                 # RAG logic & pipeline orchestration
│   ├── document_status.py             # Document processing status tracking
│   └── components/                    # Custom Haystack components
│       ├── __init__.py
│       ├── dots_ocr_converter.py      # DotsOCR PDF converter component
│       └── image_metadata_enricher.py # Image metadata enrichment
│
├── scripts/                            # Deployment scripts
│   ├── setup_environment.sh           # Environment setup
│   ├── start_vllm_server.sh          # Start DotsOCR vLLM server
│   └── start_api_server.sh           # Start FastAPI server
│
├── ui/                                 # User interface
│   ├── streamlit_app.py               # Streamlit demo UI
│   └── .streamlit/                    # Streamlit config
│
├── tests/                              # Test files
│   ├── test_simple_display.py         # Simple display tests
│   └── test_multimodal.py            # Multimodal RAG tests
│
├── notebooks/                          # Jupyter notebooks for exploration
│
├── sample_pdfs/                        # Sample PDF files for testing
│
├── dots.ocr/                          # DotsOCR VLM submodule
│   ├── demo/                          # Demo scripts
│   │   ├── demo_vllm.py              # vLLM inference demo
│   │   ├── demo_gradio.py            # Gradio UI demo
│   │   ├── demo_streamlit.py         # Streamlit UI demo
│   │   └── launch_model_vllm.sh      # Launch script
│   ├── dots_ocr/                     # Core DotsOCR package
│   │   ├── parser.py                 # Document parser
│   │   ├── model/                    # Model implementation
│   │   │   └── inference.py          # Inference logic
│   │   └── utils/                    # Utility functions
│   │       ├── image_utils.py
│   │       ├── doc_utils.py
│   │       ├── layout_utils.py
│   │       ├── format_transformer.py
│   │       └── prompts.py
│   ├── weights/                       # Model weights directory
│   │   └── DotsOCR/                  # DotsOCR model files (downloaded)
│   │       ├── config.json
│   │       ├── modeling_dots_ocr.py
│   │       ├── modeling_dots_ocr_vllm.py
│   │       └── ...
│   └── tools/
│       └── download_model.py         # Model download script
│
├── data/                              # User data (auto-created)
│   └── {user_id}/                    # Per-user isolated storage
│       ├── uploads/                  # Original uploaded PDFs
│       ├── images/                   # Extracted images from PDFs
│       └── vector_store/             # Qdrant vector database
│           └── meta.json             # Metadata
│
└── output/                            # OCR output files (optional)
    └── {doc_name}/                   # Per-document OCR results
        ├── {doc}_page_*.json         # JSON format
        └── {doc}_page_*.md           # Markdown format
```

---

### Environment Variables

```bash
# vLLM Server
export VLLM_PORT=8033
export VLLM_GPU=0

# API Server
export API_PORT=8034
export DATA_DIR=./data
export EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
export VLLM_SERVER_URL=http://localhost:8033
```


## Important Notes

### vLLM Version Requirement

⚠️ **DotsOCR requires vLLM v0.11.0+**

Earlier versions (0.6.x, 0.9.x, 0.10.x) do NOT support DotsOCR and will fail with:
```
ValueError: Model architectures ['DotsOCRForCausalLM'] are not supported
```

### Model Weights

The DotsOCR model weights (~5.8GB) must be downloaded before first run:
```bash
cd dots.ocr
python3 tools/download_model.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/users/{user_id}/documents/upload` | Upload PDFs |
| GET | `/api/users/{user_id}/documents` | List uploaded PDFs |
| DELETE | `/api/users/{user_id}/documents` | Delete specific PDFs and their chunks |
| POST | `/api/users/{user_id}/documents/reindex` | Reindex all user PDFs |
| POST | `/api/users/{user_id}/search` | Search documents |
| GET | `/api/users/{user_id}/stats` | Get document chunk count |
| GET | `/api/users/{user_id}/images` | List extracted images |
| GET | `/api/users/{user_id}/images/{filename}` | Get specific image |

Interactive API docs available at: `http://localhost:8034/docs`

---


### Testing

```bash
# Test document upload
curl -X POST "http://localhost:8034/api/users/test/documents/upload" \
  -F "files=@test.pdf"

# Test search
curl -X POST "http://localhost:8034/api/users/test/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 3}'
```

---

## Production Deployment

For production use:

1. **Add Authentication** - Implement JWT or API key authentication
2. **Use HTTPS** - Add nginx reverse proxy with SSL
3. **Add Monitoring** - Prometheus + Grafana
4. **Scale API** - Multiple API replicas behind load balancer
5. **Backup Data** - Regular backups of `./data/` directory


---


## Acknowledgments

- [DotsOCR](https://github.com/ucaslcl/DotsOCR) - Vision-Language Model for OCR
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Haystack](https://haystack.deepset.ai/) - NLP framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework

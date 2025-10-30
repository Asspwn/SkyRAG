from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os
import sys
from pathlib import Path
import uuid
import threading
from rag_service import UserRAGService
from document_status import status_tracker
from datetime import datetime

# Add dots.ocr to Python path
dots_ocr_path = os.path.abspath("./dots.ocr")
if dots_ocr_path not in sys.path:
    sys.path.insert(0, dots_ocr_path)

# Initialize FastAPI
app = FastAPI(title="Multi-User RAG API", version="1.0.0")

# Initialize RAG service with new data structure
rag_service = UserRAGService(
    storage_path="./data",  # All user data stored here
    embedding_model="intfloat/multilingual-e5-large-instruct",
    device="cuda",  # Change to "cpu" if no GPU
    vllm_server_url=os.getenv("VLLM_SERVER_URL", "http://localhost:8033"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
)

# Base data directory (user-specific subdirectories created on demand)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict
    images: List[str] = []  # List of image file paths


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.7
    include_images: bool = True  # Enable multimodal vision by default


class Source(BaseModel):
    document_number: int
    content: str
    score: float
    metadata: dict
    images: List[str] = []


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    question: str
    model: str
    retrieved_documents: int
    images_analyzed: int
    total_images_available: int


class DeleteRequest(BaseModel):
    filenames: List[str]


def _index_documents_background(user_id: str, pdf_paths: List[str], document_ids: List[str]):
    """Background task to index documents"""
    try:
        # Create metadata with document_ids
        meta = [{"user_id": user_id, "document_id": doc_id} for doc_id in document_ids]

        # Index documents
        result = rag_service.index_documents(
            user_id=user_id,
            pdf_paths=pdf_paths,
            meta=meta
        )

        # Update all document statuses to completed
        for doc_id in document_ids:
            status_tracker.update_status(doc_id, "completed")

    except Exception as e:
        # Update all document statuses to failed
        error_msg = str(e)
        for doc_id in document_ids:
            status_tracker.update_status(doc_id, "failed", error=error_msg)


@app.post("/api/users/{user_id}/documents/upload")
async def upload_documents(
    user_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload PDF documents for indexing. Returns immediately with document_ids.
    Check indexing status using /api/users/{user_id}/documents/{document_id}/status
    """
    try:
        # Create user-specific directories
        user_data_path = DATA_DIR / user_id
        user_upload_dir = user_data_path / "uploads"
        user_image_dir = user_data_path / "images"

        user_upload_dir.mkdir(parents=True, exist_ok=True)
        user_image_dir.mkdir(parents=True, exist_ok=True)

        # Generate document_ids and save files
        pdf_paths = []
        document_ids = []
        documents_info = []

        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(400, f"File {file.filename} is not a PDF")

            # Generate unique document_id
            filename = Path(file.filename).stem
            doc_uuid = str(uuid.uuid4())[:8]
            document_id = f"{user_id}_{filename}_{doc_uuid}"

            file_path = user_upload_dir / file.filename

            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            pdf_paths.append(str(file_path))
            document_ids.append(document_id)

            # Create status entry
            status_tracker.create_document(document_id, user_id, file.filename)

            documents_info.append({
                "document_id": document_id,
                "filename": file.filename,
                "status": "indexing"
            })

        # Index documents synchronously (wait for completion)
        try:
            # Create metadata with document_ids
            meta = [{"user_id": user_id, "document_id": doc_id} for doc_id in document_ids]

            # Index documents
            result = rag_service.index_documents(
                user_id=user_id,
                pdf_paths=pdf_paths,
                meta=meta
            )

            # Update all document statuses to completed
            for doc_id in document_ids:
                status_tracker.update_status(doc_id, "completed")

            # Update documents_info with completed status
            for doc_info in documents_info:
                doc_info["status"] = "completed"

            return JSONResponse(content={
                "status": "success",
                "message": f"Successfully indexed {len(files)} documents for user {user_id}",
                "documents": documents_info
            })

        except Exception as e:
            # Update all document statuses to failed
            error_msg = str(e)
            for doc_id in document_ids:
                status_tracker.update_status(doc_id, "failed", error=error_msg)
            raise HTTPException(500, f"Indexing failed: {error_msg}")

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/users/{user_id}/documents/{document_id}/reindex")
async def reindex_single_document(user_id: str, document_id: str):
    try:
        status_info = status_tracker.get_status(document_id)

        if not status_info:
            raise HTTPException(404, f"Document {document_id} not found")

        if status_info["user_id"] != user_id:
            raise HTTPException(403, "Access denied")

        filename = status_info["filename"]

        # Check if file exists
        user_data_path = DATA_DIR / user_id
        user_upload_dir = user_data_path / "uploads"
        file_path = user_upload_dir / filename

        if not file_path.exists():
            raise HTTPException(404, f"File {filename} not found in uploads folder")

        # Reset status to indexing
        status_tracker.update_status(document_id, "indexing", error=None)

        # Start background reindexing for this single document
        thread = threading.Thread(
            target=_index_documents_background,
            args=(user_id, [str(file_path)], [document_id])
        )
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "status": "success",
            "message": f"Started reindexing document {document_id}",
            "document": {
                "document_id": document_id,
                "filename": filename,
                "status": "indexing"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/users/{user_id}/documents/reindex")
async def reindex_all_documents(user_id: str):
    """
    Reindex all PDFs in user's uploads folder.
    Useful when PDFs are manually added to the uploads folder.
    """
    try:
        user_data_path = DATA_DIR / user_id
        user_upload_dir = user_data_path / "uploads"

        if not user_upload_dir.exists():
            raise HTTPException(404, f"No uploads folder found for user {user_id}")

        # Find all PDFs in uploads folder
        pdf_paths = [str(p) for p in user_upload_dir.glob("*.pdf")]

        if not pdf_paths:
            return JSONResponse(content={
                "status": "success",
                "message": f"No PDF files found in uploads folder for user {user_id}",
                "files_indexed": 0
            })

        # Index all documents
        result = rag_service.index_documents(
            user_id=user_id,
            pdf_paths=pdf_paths
        )

        return JSONResponse(content={
            "status": "success",
            "message": f"Reindexed {len(pdf_paths)} documents for user {user_id}",
            "files": [Path(p).name for p in pdf_paths],
            "details": result
        })

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/users/{user_id}/search", response_model=SearchResponse)
async def search_documents(
    user_id: str,
    request: SearchRequest
):
    """
    Search user's documents (returns raw chunks)
    """
    try:
        # Perform search
        documents = rag_service.search(
            user_id=user_id,
            query=request.query,
            top_k=request.top_k
        )

        # Format results with multimodal content
        results = []
        for doc in documents:
            # Extract image paths from metadata
            chunk_images = doc.meta.get('chunk_images', [])

            result = SearchResult(
                content=doc.content,
                score=doc.score,
                metadata=doc.meta,
                images=chunk_images  # Separate list of image paths
            )
            results.append(result)

        return SearchResponse(
            query=request.query,
            results=results
        )

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/users/{user_id}/ask", response_model=AskResponse)
async def ask_question(
    user_id: str,
    request: AskRequest
):
    """
    Ask a question and get an LLM-generated answer based on user's documents

    Supports multimodal RAG:
    - Retrieves relevant text chunks from documents
    - Automatically includes images from those chunks
    - Uses GPT-4o vision to analyze both text AND images
    - Returns answer with citations

    Set include_images=false to use text-only mode (faster, cheaper)
    """
    try:
        # Get LLM-generated answer with vision support
        result = rag_service.ask(
            user_id=user_id,
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            include_images=request.include_images
        )

        return AskResponse(**result)

    except ValueError as e:
        # Handle missing API key
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/users/{user_id}/documents/{document_id}/status")
async def get_document_status(user_id: str, document_id: str):
    """
    Get indexing status of a specific document.
    Status can be: indexing, completed, or failed
    """
    try:
        status_info = status_tracker.get_status(document_id)

        if not status_info:
            raise HTTPException(404, f"Document {document_id} not found")

        # Verify the document belongs to this user
        if status_info["user_id"] != user_id:
            raise HTTPException(403, "Access denied")

        return JSONResponse(content=status_info)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/users/{user_id}/documents/{document_id}/download")
async def download_document(user_id: str, document_id: str):
    """
    Download a user's document by document_id.
    Returns the original PDF file.
    """
    try:
        # Get document info from status tracker
        status_info = status_tracker.get_status(document_id)

        if not status_info:
            raise HTTPException(404, f"Document {document_id} not found")

        # Verify the document belongs to this user
        if status_info["user_id"] != user_id:
            raise HTTPException(403, "Access denied")

        filename = status_info["filename"]

        # Construct file path
        user_data_path = DATA_DIR / user_id
        user_upload_dir = user_data_path / "uploads"
        file_path = user_upload_dir / filename

        # Check if file exists
        if not file_path.exists():
            raise HTTPException(404, f"File {filename} not found")

        # Return file with original filename
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/users/{user_id}/documents")
async def list_documents(user_id: str):
    try:
        # Get all documents from status tracker (has document_id)
        user_documents_status = status_tracker.get_user_documents(user_id)

        # Get upload directory to fetch file stats
        user_data_path = DATA_DIR / user_id
        user_upload_dir = user_data_path / "uploads"

        documents = []
        for doc_status in user_documents_status:
            filename = doc_status["filename"]
            document_id = doc_status["document_id"]
            file_path = user_upload_dir / filename

            # Construct download URL
            download_url = f"/api/users/{user_id}/documents/{document_id}/download"

            # Get file stats if file exists
            if file_path.exists():
                stat = file_path.stat()
                # Convert Unix timestamp to ISO format
                modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

                documents.append({
                    "document_id": document_id,
                    "filename": filename,
                    "download_url": download_url,
                    "size_bytes": stat.st_size,
                    "modified_at": modified_at,
                    "status": doc_status["status"]
                })
            else:
                # File was deleted but status still exists
                documents.append({
                    "document_id": document_id,
                    "filename": filename,
                    "download_url": download_url,
                    "size_bytes": 0,
                    "modified_at": None,
                    "status": doc_status["status"]
                })

        return JSONResponse(content={
            "user_id": user_id,
            "documents": documents,
            "total_count": len(documents)
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/users/{user_id}/documents")
async def delete_documents(
    user_id: str,
    request: DeleteRequest
):
    """
    Delete specific PDF documents for a user.
    This will remove:
    - The physical PDF files from uploads folder
    - All chunks/embeddings from the vector store that came from these PDFs
    - Document status tracker entries
    """
    try:
        if not request.filenames:
            raise HTTPException(400, "No filenames provided")

        # Delete documents from storage and vector store
        result = rag_service.delete_documents(
            user_id=user_id,
            filenames=request.filenames
        )

        if result["status"] == "error":
            raise HTTPException(404, result["message"])

        # Also delete from status tracker
        for filename in request.filenames:
            # Find all document IDs for this filename
            user_docs = status_tracker.get_user_documents(user_id)
            for doc in user_docs:
                if doc["filename"] == filename:
                    status_tracker.delete_document(doc["document_id"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """
    Get statistics for a user's document store
    """
    try:
        document_store = rag_service._get_document_store(user_id)
        count = document_store.count_documents()

        return {
            "user_id": user_id,
            "document_count": count
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8034)
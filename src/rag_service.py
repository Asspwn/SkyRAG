from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.utils import ComponentDevice
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from components.dots_ocr_converter import DotsOCRConverter
from components.image_metadata_enricher import ImageMetadataEnricher
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from openai import OpenAI
import base64


class UserRAGService:
    """
    Multi-user RAG service using dots.ocr + Haystack + e5-large-instruct + Qdrant
    """

    def __init__(
        self,
        storage_path: str = "./data",
        embedding_model: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cuda",
        vllm_server_url: str = "http://localhost:8000",
        extract_images: bool = True,
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAG service

        Args:
            storage_path: Base path for storing all user data (uploads, images, vector_store)
            embedding_model: HuggingFace model for embeddings
            device: Device for embeddings ('cuda' or 'cpu')
            vllm_server_url: URL of dots.ocr vLLM server
            extract_images: Whether to extract images from PDFs as separate files
            openai_api_key: OpenAI API key for LLM generation
            llm_model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.storage_path = storage_path
        self.embedding_model = embedding_model
        # Convert device string to ComponentDevice
        self.device = ComponentDevice.from_str(device) if isinstance(device, str) else device
        self.vllm_server_url = vllm_server_url
        self.extract_images = extract_images
        self.llm_model = llm_model

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Cache for user pipelines
        self._indexing_pipelines = {}
        self._document_stores = {}

    def _get_user_data_path(self, user_id: str) -> Path:
        """Get base path for all user data"""
        return Path(self.storage_path) / user_id

    def _get_user_image_dir(self, user_id: str) -> Path:
        """Get user-specific image directory"""
        return self._get_user_data_path(user_id) / "images"

    def _get_document_store(self, user_id: str) -> QdrantDocumentStore:
        """Get or create document store for user"""
        if user_id not in self._document_stores:
            vector_store_path = self._get_user_data_path(user_id) / "vector_store"
            vector_store_path.mkdir(parents=True, exist_ok=True)

            # Check if old collection name exists (for backward compatibility)
            old_collection_name = f"user_{user_id}_docs"
            new_collection_name = "docs"

            # Check if we have an old collection from before restructuring
            meta_file = vector_store_path / "meta.json"
            collection_name = new_collection_name  # Default to new name

            if meta_file.exists():
                import json
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    # If old collection exists, use it for backward compatibility
                    if old_collection_name in meta.get("collections", {}):
                        collection_name = old_collection_name

            self._document_stores[user_id] = QdrantDocumentStore(
                path=str(vector_store_path),
                index=collection_name,
                embedding_dim=1024,  # e5-large-instruct dimension
                recreate_index=False,
            )

        return self._document_stores[user_id]

    def _build_indexing_pipeline(self, user_id: str) -> Pipeline:
        """Build indexing pipeline: PDF → dots.ocr → Split → Embed → Store"""

        document_store = self._get_document_store(user_id)

        # Get user-specific image directory
        user_image_dir = self._get_user_image_dir(user_id)
        user_image_dir.mkdir(parents=True, exist_ok=True)

        pipeline = Pipeline()

        # 1. Convert PDF to markdown with dots.ocr
        pipeline.add_component(
            "converter",
            DotsOCRConverter(
                num_threads=64,
                prompt_mode="prompt_layout_all_en",
                vllm_server_url=self.vllm_server_url,
                extract_images=self.extract_images,
                image_output_dir=str(user_image_dir)  # User-specific image directory
            )
        )

        # 2. Clean documents (remove extra whitespace, etc.)
        pipeline.add_component(
            "cleaner",
            DocumentCleaner()
        )

        # 3. Split into chunks
        # e5-large-instruct has 512 token limit
        pipeline.add_component(
            "splitter",
            DocumentSplitter(
                split_by="word",
                split_length=400,  # Keep under 512 tokens
                split_overlap=50
            )
        )

        # 4. Enrich chunks with image metadata
        pipeline.add_component(
            "image_enricher",
            ImageMetadataEnricher()
        )

        # 5. Create embeddings with e5-large-instruct
        pipeline.add_component(
            "embedder",
            SentenceTransformersDocumentEmbedder(
                model=self.embedding_model,
                device=self.device,
                batch_size=32,
            )
        )

        # 6. Write to vector store
        pipeline.add_component(
            "writer",
            DocumentWriter(document_store=document_store)
        )

        # Connect components
        pipeline.connect("converter.documents", "cleaner.documents")
        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "image_enricher.documents")
        pipeline.connect("image_enricher.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "writer.documents")

        return pipeline

    def _build_retrieval_pipeline(self, user_id: str, top_k: int = 5) -> Pipeline:
        """Build retrieval pipeline: Query → e5-instruct → Retrieve"""

        document_store = self._get_document_store(user_id)

        pipeline = Pipeline()

        # 1. Embed query with e5-large-instruct
        pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(
                model=self.embedding_model,
                device=self.device,
            )
        )

        # 2. Retrieve from Qdrant
        pipeline.add_component(
            "retriever",
            QdrantEmbeddingRetriever(
                document_store=document_store,
                top_k=top_k
            )
        )

        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        return pipeline

    def get_indexing_pipeline(self, user_id: str) -> Pipeline:
        """Get cached or create indexing pipeline for user"""
        if user_id not in self._indexing_pipelines:
            self._indexing_pipelines[user_id] = self._build_indexing_pipeline(user_id)
        return self._indexing_pipelines[user_id]

    def index_documents(
        self,
        user_id: str,
        pdf_paths: List[str],
        meta: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Index PDFs for a user

        Args:
            user_id: User identifier
            pdf_paths: List of PDF file paths
            meta: Optional metadata for each PDF

        Returns:
            Indexing result dictionary
        """
        pipeline = self.get_indexing_pipeline(user_id)

        # Ensure metadata includes user_id for each document
        if meta is None:
            meta = [{"user_id": user_id} for _ in pdf_paths]
        else:
            # Add user_id to existing metadata
            for m in meta:
                m["user_id"] = user_id

        result = pipeline.run({
            "converter": {
                "sources": pdf_paths,
                "meta": meta
            }
        })

        # Get documents written - might be int or list
        docs_written = result.get("writer", {}).get("documents_written", 0)
        if isinstance(docs_written, list):
            num_docs_written = len(docs_written)
        else:
            num_docs_written = docs_written if isinstance(docs_written, int) else 0

        return {
            "status": "success",
            "user_id": user_id,
            "documents_processed": len(pdf_paths),
            "documents_written": num_docs_written,
        }

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List:
        """
        Search user's documents

        Args:
            user_id: User identifier
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieved documents
        """
        # Build a new pipeline with the requested top_k
        # (not cached because top_k can vary per request)
        pipeline = self._build_retrieval_pipeline(user_id, top_k=top_k)

        # Format query with instruction for e5-large-instruct
        # This is CRITICAL for best performance
        task = "Given a user question, retrieve relevant passages from documents"
        formatted_query = f"Instruct: {task}\nQuery: {query}"

        result = pipeline.run({
            "text_embedder": {"text": formatted_query}
        })

        return result["retriever"]["documents"]

    def _encode_image_base64(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for OpenAI API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Failed to encode image {image_path}: {e}")
            return None

    def ask(
        self,
        user_id: str,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_images: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an LLM-generated answer based on retrieved documents
        Supports multimodal RAG: analyzes both text and images from documents

        Args:
            user_id: User identifier
            question: User's question
            top_k: Number of documents to retrieve for context
            temperature: LLM temperature (0.0-1.0)
            include_images: Whether to include images in the context (uses vision model)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured. Cannot generate answers.")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.search(user_id=user_id, query=question, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in your documents to answer this question.",
                "sources": [],
                "question": question,
                "images_analyzed": 0
            }

        # Step 2: Build context from retrieved documents and collect images
        context_parts = []
        sources = []
        all_images = []  # Collect unique images

        for idx, doc in enumerate(retrieved_docs, 1):
            # Add document content to context
            context_parts.append(f"[Document {idx}]\n{doc.content}")

            # Collect images from this chunk
            chunk_images = doc.meta.get('chunk_images', [])

            # Collect source information
            sources.append({
                "document_number": idx,
                "content": doc.content,
                "score": doc.score,
                "metadata": doc.meta,
                "images": chunk_images
            })

            # Add images to the list (avoid duplicates)
            for img_path in chunk_images:
                if img_path not in all_images:
                    all_images.append(img_path)

        context = "\n\n".join(context_parts)

        # Step 3: Build prompt with vision support
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context.

Instructions:
- Answer the question using ONLY the information from the provided documents and images
- If images are provided, analyze them carefully and incorporate visual information in your answer
- If the documents don't contain enough information, say so clearly
- Cite document numbers (e.g., [Document 1]) when referencing specific information
- Be concise but thorough
- If you mention specific data points or visual elements, cite the source document"""

        # Step 4: Build messages with vision support
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Build user message with text and images
        if include_images and all_images:
            # Use vision model with images
            user_content = [
                {
                    "type": "text",
                    "text": f"""Context from documents:

{context}

Question: {question}

Answer:"""
                }
            ]

            # Add images to the message (limit to first 10 to avoid token limits)
            images_to_include = all_images[:10]
            for img_path in images_to_include:
                # Check if image exists
                if os.path.exists(img_path):
                    img_base64 = self._encode_image_base64(img_path)
                    if img_base64:
                        # Determine image format from extension
                        ext = Path(img_path).suffix.lower()
                        if ext == '.png':
                            mime_type = "image/png"
                        elif ext in ['.jpg', '.jpeg']:
                            mime_type = "image/jpeg"
                        else:
                            mime_type = "image/png"  # default

                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}",
                                "detail": "high"  # Use high detail for better analysis
                            }
                        })

            messages.append({"role": "user", "content": user_content})
            images_analyzed = len([c for c in user_content if c.get("type") == "image_url"])
        else:
            # Text-only mode
            messages.append({
                "role": "user",
                "content": f"""Context from documents:

{context}

Question: {question}

Answer:"""
            })
            images_analyzed = 0

        # Step 5: Generate answer using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000  # Increased for vision responses
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "model": self.llm_model,
                "retrieved_documents": len(retrieved_docs),
                "images_analyzed": images_analyzed,
                "total_images_available": len(all_images)
            }

        except Exception as e:
            raise Exception(f"Failed to generate answer: {str(e)}")

    def delete_documents(
        self,
        user_id: str,
        filenames: List[str]
    ) -> Dict[str, Any]:
        """
        Delete specific documents for a user (removes PDF files and all their chunks from vector store)

        Args:
            user_id: User identifier
            filenames: List of PDF filenames to delete (e.g., ["1.pdf", "report.pdf"])

        Returns:
            Dictionary with deletion results
        """
        user_data_path = self._get_user_data_path(user_id)
        user_upload_dir = user_data_path / "uploads"

        if not user_upload_dir.exists():
            return {
                "status": "error",
                "message": f"No uploads directory found for user {user_id}",
                "deleted_files": [],
                "not_found": filenames,
                "chunks_deleted": 0
            }

        deleted_files = []
        not_found = []
        total_chunks_deleted = 0

        # Get document store
        try:
            document_store = self._get_document_store(user_id)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to access vector store: {str(e)}",
                "deleted_files": [],
                "not_found": filenames,
                "chunks_deleted": 0
            }

        # Delete each file and its chunks
        for filename in filenames:
            file_path = user_upload_dir / filename

            # Delete from vector store first (using 'source' metadata field)
            # Important: The path stored in metadata doesn't have './' prefix
            # So we need to normalize the path
            full_path_with_dot = str(file_path)  # e.g., './data/testuser/uploads/[13].pdf'
            full_path_without_dot = full_path_with_dot.lstrip('./')  # e.g., 'data/testuser/uploads/[13].pdf'

            try:
                # Get all documents and filter in Python
                # Note: We use Python filtering instead of Qdrant filters because
                # the Haystack-Qdrant filter syntax is unreliable
                all_docs = document_store.filter_documents()

                chunks_deleted = 0
                # Try both path formats (with and without './' prefix)
                for path_variant in [full_path_with_dot, full_path_without_dot]:
                    matching_docs = [doc for doc in all_docs if doc.meta.get('source') == path_variant]

                    if matching_docs:
                        # Delete by document IDs
                        doc_ids = [doc.id for doc in matching_docs]
                        document_store.delete_documents(document_ids=doc_ids)
                        chunks_deleted = len(doc_ids)
                        break  # Found the right format, no need to try the other

                total_chunks_deleted += chunks_deleted
            except Exception as e:
                print(f"Warning: Failed to delete chunks for {filename} from vector store: {e}")

            # Delete physical PDF file
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                deleted_files.append(filename)
            else:
                not_found.append(filename)

        return {
            "status": "success" if deleted_files else "error",
            "deleted_files": deleted_files,
            "not_found": not_found,
            "chunks_deleted": total_chunks_deleted,
            "message": f"Deleted {len(deleted_files)} PDF(s) and {total_chunks_deleted} chunk(s) from vector store"
        }

    def list_documents(self, user_id: str) -> Dict[str, Any]:
        """
        List all uploaded PDF documents for a user

        Args:
            user_id: User identifier

        Returns:
            Dictionary with list of documents and metadata
        """
        user_data_path = self._get_user_data_path(user_id)
        user_upload_dir = user_data_path / "uploads"

        if not user_upload_dir.exists():
            return {
                "user_id": user_id,
                "documents": [],
                "total_count": 0
            }

        documents = []
        for pdf_file in sorted(user_upload_dir.glob("*.pdf")):
            stat = pdf_file.stat()
            documents.append({
                "filename": pdf_file.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime
            })

        return {
            "user_id": user_id,
            "documents": documents,
            "total_count": len(documents)
        }
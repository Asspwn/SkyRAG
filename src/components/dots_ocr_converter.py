from haystack import component, Document, default_from_dict, default_to_dict
from typing import List, Optional, Dict, Any
import os
import tempfile
from pathlib import Path
import re
import base64
import hashlib
import uuid

@component
class DotsOCRConverter:
    """
    Haystack component that converts PDFs to markdown using dots.ocr

    Supports:
    - Text extraction (including from scanned/image-based PDFs)
    - Table parsing
    - Formula extraction (LaTeX format)
    - Multilingual content (100+ languages)

    Note: Embedded images/pictures are detected but not described
    """

    def __init__(
        self,
        num_threads: int = 64,
        prompt_mode: str = "prompt_layout_all_en",
        use_tikz_preprocess: bool = True,
        vllm_server_url: str = "http://localhost:8000",
        extract_images: bool = True,
        image_output_dir: str = "./extracted_images"
    ):
        """
        Initialize DotsOCRConverter

        Args:
            num_threads: Number of threads for multi-page PDF processing
            prompt_mode: Parsing mode ('prompt_layout_all_en', 'prompt_ocr', 'prompt_layout_only_en')
            use_tikz_preprocess: Whether to upsample images to DPI 200
            vllm_server_url: URL of the vLLM server running dots.ocr
            extract_images: Whether to extract base64 images and save them as separate files
            image_output_dir: Directory to save extracted images
        """
        self.num_threads = num_threads
        self.prompt_mode = prompt_mode
        self.use_tikz_preprocess = use_tikz_preprocess
        self.vllm_server_url = vllm_server_url
        self.extract_images = extract_images
        self.image_output_dir = image_output_dir

        # Create image output directory if it doesn't exist
        if self.extract_images:
            os.makedirs(self.image_output_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary"""
        return default_to_dict(
            self,
            num_threads=self.num_threads,
            prompt_mode=self.prompt_mode,
            use_tikz_preprocess=self.use_tikz_preprocess,
            vllm_server_url=self.vllm_server_url,
            extract_images=self.extract_images,
            image_output_dir=self.image_output_dir,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DotsOCRConverter":
        """Deserialize component from dictionary"""
        return default_from_dict(cls, data)

    def _extract_images_from_markdown(self, markdown_content: str, doc_name: str) -> tuple[str, List[str]]:
        """
        Extract base64 images from markdown and save them as separate files.

        Args:
            markdown_content: Markdown text with base64-encoded images
            doc_name: Document name for creating unique image filenames

        Returns:
            Tuple of (cleaned_markdown, list_of_image_paths)
        """
        image_paths = []

        # Pattern to match markdown images with base64 data
        # Format: ![alt_text](data:image/format;base64,DATA)
        pattern = r'!\[(.*?)\]\(data:image/(.*?);base64,(.*?)\)'

        def replace_image(match):
            alt_text = match.group(1)
            image_format = match.group(2)  # png, jpeg, etc.
            base64_data = match.group(3)

            try:
                # Decode base64 image
                image_data = base64.b64decode(base64_data)

                # Create unique filename using hash of image data
                image_hash = hashlib.md5(image_data).hexdigest()[:12]

                # Create filename: docname_hash.format
                safe_doc_name = re.sub(r'[^\w\-.]', '_', doc_name)
                image_filename = f"{safe_doc_name}_{image_hash}.{image_format}"
                image_path = os.path.join(self.image_output_dir, image_filename)

                # Save image to file
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_data)

                image_paths.append(image_path)

                # Replace with reference to saved image
                # Keep alt text if provided, otherwise use generic description
                alt = alt_text if alt_text else f"Image {len(image_paths)}"
                return f"![{alt}](image_ref:{image_path})"

            except Exception as e:
                print(f"Warning: Failed to extract image: {str(e)}")
                # Return original if extraction fails
                return match.group(0)

        # Replace all base64 images with references
        cleaned_markdown = re.sub(pattern, replace_image, markdown_content)

        return cleaned_markdown, image_paths

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[str],
        meta: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Convert PDFs to Haystack Documents using dots.ocr

        Args:
            sources: List of PDF file paths
            meta: Optional list of metadata dictionaries for each source

        Returns:
            Dictionary with 'documents' key containing converted documents
        """
        from dots_ocr.parser import DotsOCRParser
        from urllib.parse import urlparse

        # Parse vLLM server URL
        parsed = urlparse(self.vllm_server_url)
        protocol = parsed.scheme or "http"
        ip = parsed.hostname or "localhost"
        port = parsed.port or 8000

        # Initialize parser with correct parameters
        parser = DotsOCRParser(
            protocol=protocol,
            ip=ip,
            port=port,
            num_thread=self.num_threads
        )

        documents = []

        for idx, source in enumerate(sources):
            try:
                # Parse PDF with dots.ocr - returns list of page results
                results = parser.parse_file(
                    input_path=source,
                    prompt_mode=self.prompt_mode
                )

                # Combine markdown from all pages
                markdown_pages = []
                for page_result in results:
                    # Read markdown from file if available
                    if 'md_content_path' in page_result:
                        md_path = page_result['md_content_path']
                        if os.path.exists(md_path):
                            with open(md_path, 'r', encoding='utf-8') as f:
                                markdown_pages.append(f.read())

                # Combine all pages
                markdown_content = '\n\n---\n\n'.join(markdown_pages)

                # Extract images if enabled
                extracted_images = []
                if self.extract_images:
                    doc_name = Path(source).stem  # filename without extension
                    markdown_content, extracted_images = self._extract_images_from_markdown(
                        markdown_content, doc_name
                    )

                # Get metadata
                doc_meta = meta[idx] if meta and idx < len(meta) else {}

                # Extract user_id from metadata if provided
                user_id = doc_meta.get('user_id', '')

                # Use pre-generated document_id if provided, otherwise generate one
                if 'document_id' in doc_meta:
                    doc_id = doc_meta['document_id']
                else:
                    # Generate unique UUID for this upload
                    doc_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
                    filename = Path(source).stem  # filename without extension
                    if user_id:
                        doc_id = f"{user_id}_{filename}_{doc_uuid}"
                    else:
                        doc_id = f"{filename}_{doc_uuid}"

                doc_meta.update({
                    'source': source,
                    'parser': 'dots.ocr',
                    'file_name': Path(source).name,
                    'prompt_mode': self.prompt_mode,
                    'num_pages': len(results),
                    'extracted_images': extracted_images,  # List of image file paths
                    'has_images': len(extracted_images) > 0,
                    'document_id': doc_id  # Store document_id in metadata
                })

                # Create Haystack Document with custom ID
                doc = Document(
                    id=doc_id,
                    content=markdown_content,
                    meta=doc_meta
                )
                documents.append(doc)

            except Exception as e:
                print(f"Error parsing {source}: {str(e)}")
                # Optionally create error document
                error_doc = Document(
                    content="",
                    meta={
                        'source': source,
                        'error': str(e),
                        'parser': 'dots.ocr',
                        'parse_failed': True
                    }
                )
                documents.append(error_doc)

        return {"documents": documents}
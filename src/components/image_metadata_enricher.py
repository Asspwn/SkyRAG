from haystack import component, Document
from typing import List, Dict, Any
import re


@component
class ImageMetadataEnricher:
    """
    Enriches document chunks with image references based on content.

    After document splitting, this component identifies which chunks contain
    image references and updates their metadata accordingly.
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Process documents to extract image references from content and add to metadata.

        Args:
            documents: List of document chunks

        Returns:
            Dictionary with 'documents' key containing enriched documents
        """
        enriched_docs = []

        for doc in documents:
            # Find image references in content
            # Pattern: ![alt_text](image_ref:path/to/image.png)
            pattern = r'!\[.*?\]\(image_ref:(.*?)\)'
            image_refs = re.findall(pattern, doc.content)

            # Update metadata with chunk-specific image references
            chunk_meta = doc.meta.copy()

            if image_refs:
                chunk_meta['chunk_images'] = image_refs
                chunk_meta['has_chunk_images'] = True
            else:
                chunk_meta['chunk_images'] = []
                chunk_meta['has_chunk_images'] = False

            # Create new document with updated metadata
            enriched_doc = Document(
                content=doc.content,
                meta=chunk_meta,
                embedding=doc.embedding,
                id=doc.id,
                score=doc.score
            )
            enriched_docs.append(enriched_doc)

        return {"documents": enriched_docs}

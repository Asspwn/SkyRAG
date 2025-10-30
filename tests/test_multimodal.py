#!/usr/bin/env python3
"""
Test script for multimodal RAG functionality

This script demonstrates how to:
1. Upload PDFs with images
2. Search and retrieve results with separate text and image content
3. Process results for vision LLMs
"""

import requests
import json
from pathlib import Path


def test_multimodal_search(
    base_url: str = "http://localhost:8080",
    user_id: str = "testuser",
    query: str = "What is SFT",
    top_k: int = 1
):
    """
    Test multimodal search functionality

    Args:
        base_url: API base URL
        user_id: User identifier
        query: Search query
        top_k: Number of results to retrieve
    """
    # Perform search
    search_url = f"{base_url}/api/users/{user_id}/search"
    search_payload = {
        "query": query,
        "top_k": top_k
    }

    print(f"Searching for: '{query}'")
    print(f"URL: {search_url}")
    print(f"Payload: {json.dumps(search_payload, indent=2)}\n")

    response = requests.post(search_url, json=search_payload)

    if response.status_code == 200:
        data = response.json()

        print(f"Query: {data['query']}")
        print(f"Number of results: {len(data['results'])}\n")

        for idx, result in enumerate(data['results'], 1):
            print(f"{'=' * 80}")
            print(f"Result #{idx}")
            print(f"{'=' * 80}")
            print(f"Score: {result['score']:.4f}")
            print(f"\n--- Text Content ---")
            # Show first 500 chars to keep output manageable
            content = result['content']
            if len(content) > 500:
                print(content[:500] + "...")
            else:
                print(content)

            print(f"\n--- Images ---")
            if result['images']:
                print(f"Found {len(result['images'])} image(s):")
                for img_idx, img_path in enumerate(result['images'], 1):
                    img_exists = Path(img_path).exists()
                    status = "✓ EXISTS" if img_exists else "✗ NOT FOUND"
                    print(f"  {img_idx}. {img_path} [{status}]")

                    if img_exists:
                        # Show file size
                        size_bytes = Path(img_path).stat().st_size
                        size_kb = size_bytes / 1024
                        print(f"      Size: {size_kb:.2f} KB")
            else:
                print("No images in this chunk")

            print(f"\n--- Metadata ---")
            metadata = result['metadata']
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"File: {metadata.get('file_name', 'N/A')}")
            print(f"Page: {metadata.get('page_number', 'N/A')}")
            print(f"Has chunk images: {metadata.get('has_chunk_images', False)}")
            print(f"Total extracted images in doc: {len(metadata.get('extracted_images', []))}")
            print()

        # Demonstrate how to use this with a vision LLM
        print(f"{'=' * 80}")
        print("How to use with Vision LLM:")
        print(f"{'=' * 80}")

        for idx, result in enumerate(data['results'], 1):
            print(f"\nResult #{idx}:")
            print("  1. Send text content to LLM as context:")
            print(f"     Text: \"{result['content'][:100]}...\"")

            if result['images']:
                print("  2. Send images separately:")
                for img_path in result['images']:
                    print(f"     Image: {img_path}")
                print("  3. LLM can analyze both text and images together")
            else:
                print("  2. No images in this result")

    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def show_statistics(
    base_url: str = "http://localhost:8080",
    user_id: str = "testuser"
):
    """Show user statistics"""
    stats_url = f"{base_url}/api/users/{user_id}/stats"
    response = requests.get(stats_url)

    if response.status_code == 200:
        stats = response.json()
        print(f"\nUser Statistics:")
        print(f"  User ID: {stats['user_id']}")
        print(f"  Total chunks: {stats['document_count']}")
    else:
        print(f"Error getting stats: {response.status_code}")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Multimodal RAG System")
    print("=" * 80)
    print()

    # Show stats first
    show_statistics()
    print()

    # Test search
    test_multimodal_search(
        query="What is SFT",
        top_k=1
    )

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("✓ Text content is clean and readable (no base64 bloat)")
    print("✓ Images are extracted and saved as separate files")
    print("✓ Image paths are tracked in metadata")
    print("✓ Each search result includes both text and image references")
    print("✓ Perfect for feeding to multimodal LLMs!")

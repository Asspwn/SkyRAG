import streamlit as st
import requests
from pathlib import Path
from PIL import Image
import io
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="SkyRAG - Multimodal Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8034")
except:
    API_BASE_URL = "http://localhost:8034"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "documents" not in st.session_state:
    st.session_state.documents = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .stat-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def upload_documents(files, user_id):
    """Upload documents to API"""
    url = f"{API_BASE_URL}/api/users/{user_id}/documents/upload"
    files_data = [("files", (f.name, f, "application/pdf")) for f in files]
    try:
        response = requests.post(url, files=files_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def list_documents(user_id):
    """List all documents for user"""
    url = f"{API_BASE_URL}/api/users/{user_id}/documents"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to list documents: {str(e)}")
        return None

def delete_documents(user_id, filenames):
    """Delete documents"""
    url = f"{API_BASE_URL}/api/users/{user_id}/documents"
    try:
        response = requests.delete(url, json={"filenames": filenames})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Delete failed: {str(e)}")
        return None

def search_documents(user_id, query, top_k=5):
    """Search documents (raw chunks)"""
    url = f"{API_BASE_URL}/api/users/{user_id}/search"
    try:
        response = requests.post(url, json={"query": query, "top_k": top_k})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None

def ask_question(user_id, question, top_k=5, temperature=0.7, include_images=True):
    """Ask question and get LLM answer"""
    url = f"{API_BASE_URL}/api/users/{user_id}/ask"
    try:
        response = requests.post(url, json={
            "question": question,
            "top_k": top_k,
            "temperature": temperature,
            "include_images": include_images
        })
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Question failed: {str(e)}")
        return None

def get_user_stats(user_id):
    """Get user statistics"""
    url = f"{API_BASE_URL}/api/users/{user_id}/stats"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def format_file_size(bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

# Sidebar
with st.sidebar:
    st.markdown('<p class="main-header">🤖 SkyRAG</p>', unsafe_allow_html=True)
    st.markdown("**Multimodal Document Q&A**")
    st.divider()

    # User selector
    st.subheader("👤 User")
    user_id = st.text_input("User ID", value=st.session_state.user_id, key="user_input")
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Document Management
    st.subheader("📁 Document Manager")

    # Upload documents
    with st.expander("📤 Upload Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if st.button("Upload", disabled=not uploaded_files):
            with st.spinner("Uploading and indexing documents..."):
                result = upload_documents(uploaded_files, st.session_state.user_id)
                if result:
                    st.success(f"✅ Uploaded {len(uploaded_files)} document(s)")
                    st.rerun()

    # List documents
    docs_data = list_documents(st.session_state.user_id)
    if docs_data:
        st.session_state.documents = docs_data.get("documents", [])

        if st.session_state.documents:
            st.write(f"**{len(st.session_state.documents)} document(s)**")

            for doc in st.session_state.documents:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        status_icon = "✅" if doc.get("status") == "completed" else "⏳"
                        st.write(f"{status_icon} {doc['filename']}")
                        st.caption(f"{format_file_size(doc.get('size_bytes', 0))}")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['filename']}"):
                            with st.spinner("Deleting..."):
                                result = delete_documents(st.session_state.user_id, [doc['filename']])
                                if result:
                                    # Clear the cached documents to force reload
                                    st.session_state.documents = []
                                    st.rerun()
        else:
            st.info("No documents uploaded yet")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")

    top_k = st.slider("Number of sources (top-k)", 1, 10, 5, key="top_k")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature")
    include_images = st.checkbox("Enable Vision (analyze images)", value=True, key="include_images")

    st.caption("💡 Vision mode uses GPT-4o to analyze images")

    st.divider()

    # Stats
    stats = get_user_stats(st.session_state.user_id)
    if stats:
        st.subheader("📊 Statistics")
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats.get('document_count', 0)}</div>
            <div class="stat-label">Indexed Chunks</div>
        </div>
        """, unsafe_allow_html=True)

# Main content
st.markdown('<p class="main-header">💬 Chat with Your Documents</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Search"])

with tab1:
    # Chat interface
    if not st.session_state.documents:
        st.warning("⚠️ No documents uploaded yet. Please upload documents in the sidebar to get started.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if "sources" in message:
                with st.expander("📚 View Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Document {source['document_number']}</strong>
                            (Score: {source['score']:.3f})<br>
                            {source['content'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)

                        # Display images if available
                        if source.get("images"):
                            st.write("**Images from this source:**")
                            img_cols = st.columns(min(len(source["images"]), 3))
                            for idx, img_path in enumerate(source["images"][:3]):
                                try:
                                    with img_cols[idx % 3]:
                                        # Try to load and display image
                                        full_path = Path("..") / img_path
                                        if full_path.exists():
                                            img = Image.open(full_path)
                                            st.image(img, use_container_width=True)
                                        else:
                                            st.caption(f"Image: {Path(img_path).name}")
                                except:
                                    st.caption(f"Image: {Path(img_path).name}")

            # Display metadata
            if "metadata" in message:
                meta = message["metadata"]
                st.caption(f"🤖 Model: {meta.get('model', 'N/A')} | "
                          f"📄 Sources: {meta.get('retrieved_documents', 0)} | "
                          f"🖼️ Images: {meta.get('images_analyzed', 0)}/{meta.get('total_images_available', 0)}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(
                    st.session_state.user_id,
                    prompt,
                    top_k=st.session_state.top_k,
                    temperature=st.session_state.temperature,
                    include_images=st.session_state.include_images
                )

                if result:
                    answer = result["answer"]
                    st.markdown(answer)

                    # Store assistant message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result.get("sources", []),
                        "metadata": {
                            "model": result.get("model"),
                            "retrieved_documents": result.get("retrieved_documents"),
                            "images_analyzed": result.get("images_analyzed"),
                            "total_images_available": result.get("total_images_available")
                        }
                    })

                    # Display sources
                    with st.expander("📚 View Sources", expanded=False):
                        for source in result.get("sources", []):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Document {source['document_number']}</strong>
                                (Score: {source['score']:.3f})<br>
                                {source['content'][:300]}...
                            </div>
                            """, unsafe_allow_html=True)

                            # Display images
                            if source.get("images"):
                                st.write("**Images from this source:**")
                                img_cols = st.columns(min(len(source["images"]), 3))
                                for idx, img_path in enumerate(source["images"][:3]):
                                    try:
                                        with img_cols[idx % 3]:
                                            full_path = Path("..") / img_path
                                            if full_path.exists():
                                                img = Image.open(full_path)
                                                st.image(img, use_container_width=True)
                                            else:
                                                st.caption(f"Image: {Path(img_path).name}")
                                    except:
                                        st.caption(f"Image: {Path(img_path).name}")

                    # Display metadata
                    st.caption(f"🤖 Model: {result.get('model')} | "
                              f"📄 Sources: {result.get('retrieved_documents')} | "
                              f"🖼️ Images: {result.get('images_analyzed')}/{result.get('total_images_available')}")

                    st.rerun()

with tab2:
    # Raw search interface
    st.subheader("🔍 Raw Document Search")
    st.caption("Search for raw document chunks (without LLM generation)")

    search_query = st.text_input("Search query", placeholder="Enter search terms...")
    search_top_k = st.slider("Number of results", 1, 10, 5, key="search_top_k")

    if st.button("Search", disabled=not search_query):
        with st.spinner("Searching..."):
            results = search_documents(st.session_state.user_id, search_query, search_top_k)

            if results:
                st.success(f"Found {len(results.get('results', []))} results")

                for idx, result in enumerate(results.get("results", []), 1):
                    with st.expander(f"Result {idx} (Score: {result['score']:.3f})", expanded=True):
                        st.markdown(result["content"])

                        # Metadata
                        st.caption(f"**Source:** {result['metadata'].get('file_name', 'N/A')} | "
                                  f"**Page:** {result['metadata'].get('page_number', 'N/A')}")

                        # Images
                        if result.get("images"):
                            st.write("**Associated images:**")
                            img_cols = st.columns(min(len(result["images"]), 3))
                            for idx, img_path in enumerate(result["images"][:3]):
                                try:
                                    with img_cols[idx % 3]:
                                        full_path = Path("..") / img_path
                                        if full_path.exists():
                                            img = Image.open(full_path)
                                            st.image(img, use_container_width=True)
                                        else:
                                            st.caption(f"Image: {Path(img_path).name}")
                                except:
                                    st.caption(f"Image: {Path(img_path).name}")

# Footer
st.divider()
st.caption("🚀 Powered by SkyRAG - DotsOCR + GPT-4o + Qdrant")

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

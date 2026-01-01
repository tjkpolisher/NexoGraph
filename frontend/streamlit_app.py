"""Nexograph Streamlit MVP UI.

This module provides a simple web interface for the Nexograph knowledge graph system,
allowing users to upload documents and interact with the Q&A chatbot.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import streamlit as st

# ============================================================================
# Configuration
# ============================================================================

# Backend API configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_TIMEOUT = 120.0  # 2 minutes for document processing

# Page configuration
st.set_page_config(
    page_title="Nexograph",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }

    /* Upload section styling */
    .upload-section {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Document card styling */
    .document-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "last_doc_refresh" not in st.session_state:
        st.session_state.last_doc_refresh = None


# ============================================================================
# API Helper Functions
# ============================================================================

async def check_api_health() -> bool:
    """Check if backend API is healthy.

    Returns:
        True if API is healthy, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/health")
            return response.status_code == 200
    except Exception:
        return False


async def upload_document(
    file_content: bytes,
    filename: str,
    title: Optional[str] = None,
    category: str = "paper",
    tags: Optional[str] = None,
) -> dict:
    """Upload a document to the backend.

    Args:
        file_content: File binary content
        filename: Original filename
        title: Optional document title
        category: Document category
        tags: Comma-separated tags

    Returns:
        API response dictionary

    Raises:
        httpx.HTTPError: If upload fails
    """
    files = {"file": (filename, file_content)}
    data = {"category": category}

    if title:
        data["title"] = title
    if tags:
        data["tags"] = tags

    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        response = await client.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return response.json()


async def get_documents(limit: int = 10) -> list[dict]:
    """Get list of documents from backend.

    Args:
        limit: Maximum number of documents to retrieve

    Returns:
        List of document dictionaries

    Raises:
        httpx.HTTPError: If request fails
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{API_BASE_URL}/documents",
            params={"page": 1, "limit": limit},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("documents", [])


async def chat_query(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    include_sources: bool = True,
) -> dict:
    """Send a chat query to the backend.

    Args:
        query: User's question
        mode: Search mode (local/global/hybrid/naive)
        top_k: Number of chunks to retrieve
        include_sources: Whether to include source information

    Returns:
        API response with answer and sources

    Raises:
        httpx.HTTPError: If request fails
    """
    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "include_sources": include_sources,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def delete_document(document_id: str) -> None:
    """Delete a document from the backend.

    Args:
        document_id: Document UUID to delete

    Raises:
        httpx.HTTPError: If deletion fails
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.delete(
            f"{API_BASE_URL}/documents/{document_id}"
        )
        response.raise_for_status()


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render the sidebar with document upload and list."""
    with st.sidebar:
        # Title and description
        st.title("🔗 Nexograph")
        st.markdown("*AI 지식 베이스 시스템*")
        st.divider()

        # Document upload section
        st.subheader("📤 문서 업로드")

        with st.container():
            uploaded_file = st.file_uploader(
                "PDF 또는 Markdown 파일 업로드",
                type=["pdf", "md"],
                help="최대 파일 크기: 50MB",
            )

            if uploaded_file is not None:
                # Document metadata inputs
                title = st.text_input(
                    "제목 (선택사항)",
                    placeholder="비워두면 파일명을 사용합니다",
                )

                category = st.selectbox(
                    "카테고리",
                    options=["paper", "blog", "documentation"],
                    index=0,
                )

                tags = st.text_input(
                    "태그 (쉼표로 구분)",
                    placeholder="예: AI, NLP, Transformer",
                )

                # Upload button
                if st.button("📥 업로드", type="primary", use_container_width=True):
                    with st.spinner("문서를 처리하는 중..."):
                        try:
                            # Read file content
                            file_content = uploaded_file.read()

                            # Upload document
                            result = asyncio.run(upload_document(
                                file_content=file_content,
                                filename=uploaded_file.name,
                                title=title if title else None,
                                category=category,
                                tags=tags if tags else None,
                            ))

                            st.success(f"✅ 문서가 성공적으로 업로드되었습니다!")
                            st.info(f"""
                            **문서 ID**: {result['document_id']}
                            **청크 수**: {result.get('chunks_count', 'N/A')}
                            **처리 시간**: {result.get('processing_time_ms', 'N/A')}ms
                            """)

                            # Refresh document list
                            st.session_state.last_doc_refresh = None
                            st.rerun()

                        except httpx.HTTPError as e:
                            st.error(f"❌ 업로드 실패: {e}")
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {e}")

        st.divider()

        # Document list section
        st.subheader("📚 문서 목록")

        # Refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄", help="문서 목록 새로고침"):
                st.session_state.last_doc_refresh = None
                st.rerun()

        # Fetch documents (with caching)
        if (st.session_state.last_doc_refresh is None or
            (datetime.now() - st.session_state.last_doc_refresh).seconds > 60):
            try:
                with st.spinner("문서 목록을 불러오는 중..."):
                    docs = asyncio.run(get_documents(limit=10))
                    st.session_state.documents = docs
                    st.session_state.last_doc_refresh = datetime.now()
            except Exception as e:
                st.error(f"❌ 문서 목록 로드 실패: {e}")
                st.session_state.documents = []

        # Display documents
        if st.session_state.documents:
            for doc in st.session_state.documents:
                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"""
                        <div class="document-card">
                            <strong>{doc['title'][:40]}{'...' if len(doc['title']) > 40 else ''}</strong><br>
                            <small>📁 {doc['category']} | 🧩 {doc['chunks_count']} chunks</small><br>
                            <small>🏷️ {', '.join(doc.get('tags', [])[:3])}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        if st.button("🗑️", key=f"delete_{doc['id']}", help="삭제"):
                            try:
                                asyncio.run(delete_document(doc['id']))
                                st.success("삭제 완료!")
                                st.session_state.last_doc_refresh = None
                                st.rerun()
                            except Exception as e:
                                st.error(f"삭제 실패: {e}")
        else:
            st.info("📭 업로드된 문서가 없습니다.")

        # Footer
        st.divider()
        st.caption("Nexograph v0.1.0 | Phase 1 MVP")


def render_main_chat():
    """Render the main chat interface."""
    st.title("💬 AI 지식 베이스 Q&A")
    st.markdown("문서 기반 질문답변 시스템입니다. 무엇이든 물어보세요!")

    # Search mode selection
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_mode = st.radio(
            "검색 모드",
            options=["hybrid", "local", "global", "naive"],
            index=0,
            horizontal=True,
            help="""
            - **hybrid**: 벡터 검색 + 그래프 검색 (권장)
            - **local**: 엔티티 중심 검색
            - **global**: 전역 지식 검색
            - **naive**: 단순 벡터 검색
            """,
        )

    with col2:
        top_k = st.number_input(
            "검색 청크 수",
            min_value=1,
            max_value=20,
            value=5,
            help="검색할 문서 청크 개수",
        )

    with col3:
        include_sources = st.checkbox(
            "출처 표시",
            value=True,
            help="답변과 함께 출처 정보 표시",
        )

    st.divider()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander(f"📚 출처 ({len(sources)}개 문서)"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            **{i}. {source['document_title']}**
                            (유사도: {source['relevance_score']:.2f})

                            > {source['chunk_preview']}

                            ---
                            """)

    # Chat input
    if prompt := st.chat_input("질문을 입력하세요..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                try:
                    # Query API
                    response = asyncio.run(chat_query(
                        query=prompt,
                        mode=search_mode,
                        top_k=int(top_k),
                        include_sources=include_sources,
                    ))

                    answer = response["answer"]
                    sources = response.get("sources", [])
                    processing_time = response.get("processing_time_ms", 0)

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if include_sources and sources:
                        with st.expander(f"📚 출처 ({len(sources)}개 문서)"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                **{i}. {source['document_title']}**
                                (유사도: {source['relevance_score']:.2f})

                                > {source['chunk_preview']}

                                ---
                                """)

                    # Display processing time
                    st.caption(f"⏱️ 처리 시간: {processing_time}ms | 모드: {search_mode}")

                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "processing_time": processing_time,
                        "mode": search_mode,
                    })

                except httpx.HTTPError as e:
                    error_msg = f"❌ API 오류: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

                except Exception as e:
                    error_msg = f"❌ 오류 발생: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


def render_api_status():
    """Render API connection status in the sidebar."""
    with st.sidebar:
        st.divider()

        # Check API health
        try:
            is_healthy = asyncio.run(check_api_health())

            if is_healthy:
                st.success("🟢 API 연결됨")
            else:
                st.error("🔴 API 연결 실패")
                st.info("백엔드 서버를 시작하세요:\n```bash\nuvicorn backend.main:app --reload\n```")
        except Exception:
            st.error("🔴 API 연결 실패")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    # Import asyncio here to avoid issues
    import asyncio

    # Make asyncio available globally
    globals()['asyncio'] = asyncio

    # Initialize session state
    init_session_state()

    # Render UI components
    render_sidebar()
    render_api_status()
    render_main_chat()

    # Add clear chat button in sidebar
    with st.sidebar:
        st.divider()
        if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()

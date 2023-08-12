import pandas as pd
import streamlit as st
from typing import Dict, List
from pythainlp.tokenize import word_tokenize
from mkr.retrievers.baseclass import Retriever
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.dense_retriever import DenseRetriever
from mkr.retrievers.sparse_retriever import BM25SparseRetriever
from mkr.retrievers.hybrid_retriever import HybridRetriever
from mkr.retrievers.document_retriever import DocumentRetriever


# Session state management
if "init" not in st.session_state:
    # Initialize session state here
    st.session_state.init = True

    st.session_state.resource_manager = ResourceManager(force_download=False)
    st.session_state.dense_retriever: Retriever = DenseRetriever.from_indexed(
        st.session_state.resource_manager.get_index_path(f"wikipedia_th_v2_mUSE")
    )
    st.session_state.sparse_retriever: Retriever = BM25SparseRetriever.from_indexed(
        st.session_state.resource_manager.get_index_path(f"wikipedia_th_v2_bm25_okapi_newmm")
    )
    st.session_state.hybrid_retriever: Retriever = HybridRetriever(
        dense_retriever=st.session_state.dense_retriever,
        sparse_retriever=st.session_state.sparse_retriever,
    )
    st.session_state.retriever: Retriever = DocumentRetriever(st.session_state.sparse_retriever)
    st.session_state.response: List[Dict[str, str]] = None
    
    st.session_state.selecting_retriever: str = "Sparse retrieval"
    st.session_state.selecting_content_level: str = "Document"
    st.session_state.sparse_weight: float = 0.5
    st.session_state.query: str = ""

# Callbacks
def on_search():
    query = st.session_state.query
    if query == "":
        return
    retriever = st.session_state.retriever

    if st.session_state.selecting_retriever == "Hybrid retrieval":
        results = retriever([query], top_k=st.session_state.top_k, sparse_weight=st.session_state.sparse_weight).resultss[0]
    else:
        results = retriever([query], top_k=st.session_state.top_k).resultss[0]
    st.session_state.response = results

def on_selecting_retriever():
    if st.session_state.selecting_retriever == "Hybrid retrieval":
        if st.session_state.selecting_content_level == "Document":
            st.session_state.hybrid_retriever.dense_retriever = DocumentRetriever(st.session_state.dense_retriever)
            st.session_state.hybrid_retriever.sparse_retriever = DocumentRetriever(st.session_state.sparse_retriever)
        elif st.session_state.selecting_content_level == "Paragraph":
            st.session_state.hybrid_retriever.dense_retriever = st.session_state.dense_retriever
            st.session_state.hybrid_retriever.sparse_retriever = st.session_state.sparse_retriever
        st.session_state.retriever = st.session_state.hybrid_retriever
    else:
        if st.session_state.selecting_retriever == "Sparse retrieval":
            retriever = st.session_state.sparse_retriever
        elif st.session_state.selecting_retriever == "Dense retrieval":
            retriever = st.session_state.dense_retriever

        if st.session_state.selecting_content_level == "Document":
            st.session_state.retriever = DocumentRetriever(retriever)
        elif st.session_state.selecting_content_level == "Paragraph":
            st.session_state.retriever = retriever

    on_search()
    st.toast(f"Retriever changed to {st.session_state.selecting_content_level}-Level {st.session_state.selecting_retriever}.")

# Frontend
st.text_input("Enter query here:", key="query", on_change=on_search)

with st.sidebar:
    st.selectbox(
        "Retriever:", 
        ["Sparse retrieval", "Dense retrieval", "Hybrid retrieval"], 
        key="selecting_retriever",
        on_change=on_selecting_retriever
    )
    if st.session_state.selecting_retriever == "Sparse retrieval":
        st.selectbox("Model:", ["BM25_okapi"], key="selecting_sparse_model", on_change=on_selecting_retriever)
    elif st.session_state.selecting_retriever == "Dense retrieval":
        st.selectbox("Model:", ["mUSE"], key="selecting_dense_model", on_change=on_selecting_retriever)
    elif st.session_state.selecting_retriever == "Hybrid retrieval":
        st.selectbox("Dense model:", ["mUSE"], key="selecting_dense_model", on_change=on_selecting_retriever)
        st.selectbox("Sparse model:", ["BM25_okapi"], key="selecting_sparse_model", on_change=on_selecting_retriever)
    st.selectbox(
        "Content-level:",
        ["Paragraph", "Document"],
        key="selecting_content_level",
        on_change=on_selecting_retriever
    )
    st.slider("Top-k:", 1, 100, 3, key="top_k", on_change=on_search)
    if st.session_state.selecting_retriever == "Hybrid retrieval":
        st.slider("Sparse weight:", 0.0, 1.0, 0.5, key="sparse_weight", on_change=on_search)

if st.session_state.response is not None:
    results = st.session_state.response
    for result in results.values():
        url = result["doc_url"]
        title = result["doc_title"]
        content = result["doc_text"]
        content = " ".join(word_tokenize(content)[:200])
        st.markdown(f"#### [{title}]({url})")
        st.markdown(content)
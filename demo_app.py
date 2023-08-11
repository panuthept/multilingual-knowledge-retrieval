import pandas as pd
import streamlit as st
from typing import Dict, List
from pythainlp.tokenize import word_tokenize
from mkr.retrievers.baseclass import Retriever
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.dense_retriever import DenseRetriever
from mkr.retrievers.sparse_retriever import BM25SparseRetriever
from mkr.retrievers.document_retriever import DocumentRetriever


# Session state management
if "init" not in st.session_state:
    # Initialize session state here
    st.session_state.init = True

    st.session_state.resource_manager = ResourceManager(force_download=False)
    st.session_state.retriever: Retriever = DenseRetriever.from_indexed(st.session_state.resource_manager.get_index_path(f"wikipedia_th_v2_mUSE"))
    st.session_state.doc_retriever: DocumentRetriever = DocumentRetriever(st.session_state.retriever)
    st.session_state.corpus = pd.read_csv(st.session_state.resource_manager.get_corpus_path("wikipedia_th_v2_raw"))
    st.session_state.response: List[Dict[str, str]] = None
    
    st.session_state.current_retriever: str = "Dense retrieval"
    st.session_state.selecting_retriever: str = "Dense retrieval"
    st.session_state.current_model: str = "mUSE"
    st.session_state.selecting_model: str = "mUSE"
    st.session_state.query: str = ""

# Callbacks
def on_search():
    query = st.session_state.query
    doc_retriever = st.session_state.doc_retriever

    results = doc_retriever([query], top_k=st.session_state.top_k)[0]
    st.session_state.response = results

def on_selecting_retriever():
    if st.session_state.selecting_retriever != st.session_state.current_retriever:
        if st.session_state.selecting_retriever == "Sparse retrieval":
            st.session_state.retriever = BM25SparseRetriever.from_indexed(st.session_state.resource_manager.get_index_path(f"wikipedia_th_v2_bm25_okapi_newmm"))
        elif st.session_state.selecting_retriever == "Dense retrieval":
            st.session_state.retriever = DenseRetriever.from_indexed(st.session_state.resource_manager.get_index_path(f"wikipedia_th_v2_mUSE"))
        st.session_state.doc_retriever = DocumentRetriever(st.session_state.retriever)
        st.session_state.current_retriever = st.session_state.selecting_retriever
        st.session_state.query = ""
        st.session_state.response = None
        st.toast(f"Retriever changed to {st.session_state.selecting_retriever}")

# Frontend
st.text_input("Enter query here:", key="query", on_change=on_search)

with st.sidebar:
    st.selectbox(
        "Retriever:", 
        ["Sparse retrieval", "Dense retrieval"], 
        key="selecting_retriever",
        on_change=on_selecting_retriever
    )
    if st.session_state.selecting_retriever == "Sparse retrieval":
        st.selectbox("Model:", ["BM25_okapi"], key="selecting_model", on_change=on_selecting_retriever)
    elif st.session_state.selecting_retriever == "Dense retrieval":
        st.selectbox("Model:", ["mUSE"], key="selecting_model", on_change=on_selecting_retriever)
    st.slider("Top-k:", 1, 100, 3, key="top_k", on_change=on_search)

if st.session_state.response is not None:
    results = st.session_state.response
    for result in results:
        url = result["url"]
        title = result["title"]
        content = st.session_state.corpus[st.session_state.corpus["title"] == title]["text"].values[0]
        content = " ".join(word_tokenize(content)[:200])
        st.markdown(f"#### [{title}]({url})")
        st.markdown(content)
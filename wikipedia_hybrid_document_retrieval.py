import os
import argparse
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.hybrid_retriever import HybridRetriever
from mkr.retrievers.document_retriever import DocumentRetriever
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig
from mkr.retrievers.sparse_retriever import BM25SparseRetriever, BM25Config


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--sparse_weight", default=0.5, type=float)
    parser.add_argument("--dense_model_name", default="mUSE")
    parser.add_argument("--sparse_model_name", default="bm25_okapi")
    parser.add_argument("--tokenizer_name", default="newmm", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    resource_manager = ResourceManager(force_download=args.force_download)
    dense_index_path = resource_manager.get_index_path(f"wikipedia_th_v2_{args.dense_model_name}", download=args.download)
    sparse_index_path = resource_manager.get_index_path(f"wikipedia_th_v2_{args.sparse_model_name}_{args.tokenizer_name}", download=args.download)
    corpus_path = resource_manager.get_corpus_path("wikipedia_th_v2", download=args.download)

    # Prepare retrievers
    if os.path.exists(dense_index_path):
        dense_retriever = DenseRetriever.from_indexed(dense_index_path)
    else:
        dense_retriever = DenseRetriever(
            config=DenseRetrieverConfig(
                model_name=args.dense_index_path,
                corpus_dir=corpus_path,
            )
        )
        dense_retriever.save_index(dense_index_path)
    dense_retriever = DocumentRetriever(dense_retriever)

    if os.path.exists(sparse_index_path):
        sparse_retriever = BM25SparseRetriever.from_indexed(sparse_index_path)
    else:
        sparse_retriever = BM25SparseRetriever(
            config=BM25Config(
                model_name=args.sparse_index_path,
                tokenizer_name=args.tokenizer_name,
                corpus_dir=corpus_path,
            )
        )
        sparse_retriever.save_index(sparse_index_path)
    sparse_retriever = DocumentRetriever(sparse_retriever)

    doc_retriever = HybridRetriever(dense_retriever, sparse_retriever)

    # Load queries
    queries = [args.query]
    output = doc_retriever(queries, top_k=args.top_k, sparse_weight=args.sparse_weight)

    # Retrieve documents
    print(f"Query: {queries[0]}")
    print("-" * 150)
    for result in output.resultss[0].values():
        for k, v in result.items():
            print(f"{k}: {v}")
        print("-" * 150)
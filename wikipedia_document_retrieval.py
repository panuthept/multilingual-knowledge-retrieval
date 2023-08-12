import os
import argparse
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.document_retriever import DocumentRetriever
from mkr.retrievers.dense_retriever import DenseRetriever, EncoderConfig


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--model_name", default="mUSE")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    resource_manager = ResourceManager(force_download=args.force_download)
    index_path = resource_manager.get_index_path(f"wikipedia_th_v2_{args.model_name}", download=args.download)
    corpus_path = resource_manager.get_corpus_path("wikipedia_th_v2", download=args.download)

    # Prepare retriever
    if os.path.exists(index_path):
        retriever = DenseRetriever.from_indexed(index_path)
    else:
        retriever = DenseRetriever(
            config=EncoderConfig(
                model_name=args.model_name,
                corpus_dir=corpus_path,
            )
        )
        retriever.save_index(index_path)
    doc_retriever = DocumentRetriever(retriever)

    # Load queries
    queries = [args.query]
    output = doc_retriever(queries, top_k=args.top_k)

    # Retrieve documents
    print(f"Query: {queries[0]}")
    print("-" * 150)
    for result in output.resultss[0].values():
        for k, v in result.items():
            print(f"{k}: {v}")
        print("-" * 150)
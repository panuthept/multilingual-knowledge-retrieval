import os
import argparse
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.dense_retriever import DenseRetriever, EncoderConfig


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--model_name", default="mUSE")
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    resource_manager = ResourceManager(force_download=args.force_download)
    index_path = resource_manager.get_index_path(f"wikipedia_th_{args.model_name}")
    corpus_path = resource_manager.get_corpus_path("wikipedia_th")

    # Prepare retriever
    if os.path.exists(index_path):
        doc_retriever = DenseRetriever.from_indexed(index_path)
    else:
        doc_retriever = DenseRetriever(
            config=EncoderConfig(
                model_name=args.model_name,
                corpus_dir=corpus_path,
            )
        )
        doc_retriever.save_index(index_path)

    # Load queries
    queries = [args.query]
    results = doc_retriever(queries, top_k=args.top_k)[0]

    # Retrieve documents
    print(f"Query: {queries[0]}")
    print("-" * 150)
    for result in results:
        for k, v in result.items():
            print(f"{k}: {v}")
        print("-" * 150)
import json
import argparse

from mkr.retrievers.dense_retriever import DenseRetriever, EncoderConfig


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str, help="Query string")
    parser.add_argument("--corpus_file", required=True, type=str, help="Document file")
    parser.add_argument("--index_file", required=True, type=str, help="Index file")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for encoding")
    parser.add_argument("--top_k", default=3, type=int, help="Retrieve top k documents")
    parser.add_argument("--model_name", default="mUSE", type=str, help="Encoder to use")
    args = parser.parse_args()

    # Prepare retriever
    # doc_retriever = DenseRetriever(
    #     config=EncoderConfig(
    #         model_name=args.model_name,
    #         corpus_dir=args.corpus_file,
    #         batch_size=args.batch_size,
    #     )
    # )
    # doc_retriever.save_index(args.index_file)
    doc_retriever = DenseRetriever.from_indexed(args.index_file)

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
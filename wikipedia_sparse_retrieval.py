import json
import argparse

from mkr.retrievers.sparse_retriever import BM25SparseRetriever, BM25Config


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str, help="Query string")
    parser.add_argument("--corpus_file", required=True, type=str, help="Document file")
    parser.add_argument("--index_file", required=True, type=str, help="Index file")
    parser.add_argument("--top_k", default=3, type=int, help="Retrieve top k documents")
    parser.add_argument("--model_name", default="bm25_okapi", type=str, help="BM25 to use")
    parser.add_argument("--tokenizer_name", default="newmm", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    # Prepare retriever
    # doc_retriever = BM25SparseRetriever(
    #     config=BM25Config(
    #         model_name=args.model_name,
    #         tokenizer_name=args.tokenizer_name,
    #         corpus_dir=args.corpus_file,
    #     )
    # )
    # doc_retriever.save_index(args.index_file)
    doc_retriever = BM25SparseRetriever.from_indexed(args.index_file)

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
import json
import argparse
from mkr.retrievers.sparse_retriever import BM25SparseRetriever, BM25Config


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", default=3, type=int, help="Retrieve top k documents")
    parser.add_argument("--model_name", default="bm25_okapi", type=str, help="BM25 to use")
    parser.add_argument("--tokenizer_name", default="newmm", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    corpus_dir = "./demo_data/demo_docs.jsonl"
    queries_dir = "./demo_data/demo_queries.jsonl"
    qrels_dir = "./demo_data/demo_qrels.jsonl"

    # Prepare retriever
    doc_retriever = BM25SparseRetriever(
        config=BM25Config(
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            corpus_dir=corpus_dir,
        )
    )

    # Load queries
    queries = []
    with open(queries_dir, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query_text"])

    # Load qrels
    qrels = {}
    with open(qrels_dir, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qrels[data["query_id"]] = data["doc_ids"]

    # Retrieve documents
    output = doc_retriever(queries, top_k=args.top_k)
    print("NOTE: [✅] means the document is relevant, [❌] means the document is not relevant.")
    for q_id, (query, results) in enumerate(zip(output.queries, output.resultss)):
        print(f"Query: {query}")
        print("-" * 150)
        for result in results.values():
            if result["doc_id"] in qrels[q_id]:
                print(f"[✅] Score: {result['score']:.4f}\tDocument[{result['doc_id']}]: {result['doc_text']}")
            else:
                print(f"[❌] Score: {result['score']:.4f}\tDocument[{result['doc_id']}]: {result['doc_text']}")
            print("-" * 150)
        print()
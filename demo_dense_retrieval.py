import json
import argparse
from mkr.retrievers.dense_retriever import DenseRetriever, EncoderConfig


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for encoding")
    parser.add_argument("--top_k", default=3, type=int, help="Retrieve top k documents")
    parser.add_argument("--model_name", default="mUSE", type=str, help="Encoder to use")
    args = parser.parse_args()

    corpus_dir = "./demo_data/demo_docs.jsonl"
    queries_dir = "./demo_data/demo_queries.jsonl"
    qrels_dir = "./demo_data/demo_qrels.jsonl"

    # Prepare retriever
    doc_retriever = DenseRetriever(
        config=EncoderConfig(
            model_name=args.model_name,
            corpus_dir=corpus_dir,
            batch_size=args.batch_size,
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
    print("NOTE: [✅] means the document is relevant, [❌] means the document is not relevant.")
    for q_id, (query, results) in enumerate(zip(queries, doc_retriever(queries, top_k=args.top_k))):
        print(f"Query: {query}")
        print("-" * 150)
        for result in results:
            if result["doc_id"] in qrels[q_id]:
                print(f"[✅] Score: {result['score']:.4f}\tDocument[{result['doc_id']}]: {result['doc_text']}")
            else:
                print(f"[❌] Score: {result['score']:.4f}\tDocument[{result['doc_id']}]: {result['doc_text']}")
            print("-" * 150)
        print()
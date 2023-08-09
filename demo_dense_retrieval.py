import json
import argparse

from mkr.retrievers.dense_retriever import DenseRetriever, EncoderConfig


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", required=True, type=str, help="Query file with questions")
    parser.add_argument("--doc_file", required=True, type=str, help="Document file with sentences to encode")
    parser.add_argument("--qrel_file", required=True, type=str, help="Query relevance file")
    parser.add_argument("--index_file", required=True, type=str, help="Index file with encoded sentences")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for encoding")
    parser.add_argument("--top_k", default=3, type=int, help="Retrieve top k documents")
    parser.add_argument("--model_name", default="mUSE", type=str, help="Encoder to use")
    args = parser.parse_args()

    # Prepare retriever
    doc_retriever = DenseRetriever(
        config=EncoderConfig(
            model_name=args.model_name,
            corpus_dir=args.doc_file,
            batch_size=args.batch_size,
        )
    )
    doc_retriever.save_index(args.index_file)
    # doc_retriever = DenseRetriever.from_indexed(args.index_file)

    # Load queries
    queries = []
    with open(args.query_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query_text"])

    # Load qrels
    qrels = {}
    with open(args.qrel_file, "r", encoding="utf-8") as f:
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
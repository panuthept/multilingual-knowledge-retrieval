import json
import argparse
from mkr.benchmark import Benchmark
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mUSE")
    args = parser.parse_args()

    # Prepare retriever
    doc_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name=args.model_name,
            database_path=f"./database/{args.model_name}",
        ),
    )
    doc_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    doc_retrieval.add_corpus("tydiqa", "./corpus/tydiqa/corpus.jsonl")

    # Prepare benchmark
    benchmark = Benchmark(doc_retrieval)

    # IAPP-WikiQA
    ####################################################################################
    print("IAPP-WikiQA Performance:")
    # Load qrels
    qrels = []
    with open("./datasets/iapp_wiki_qa/qrel_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    eval_metrics = benchmark.evaluate(
        corpus_name="iapp_wiki_qa",
        qrels=qrels,
    )

    for key, values in eval_metrics.items():
        print(f"{key}: {values * 100:.1f}")
    ####################################################################################
    # TYDI-QA (Primary)
    ####################################################################################
    print("TYDI-QA (Primary) Performance:")
    # Load qrels
    qrels = []
    with open("./datasets/tydiqa/qrel_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    eval_metrics = benchmark.evaluate(
        corpus_name="tydiqa",
        qrels=qrels,
    )

    for key, values in eval_metrics.items():
        print(f"{key}: {values * 100:.1f}")
    ####################################################################################
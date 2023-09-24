import argparse
from mkr.benchmark import Benchmark
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mUSE")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Check GPU available
    if args.model_name == "mUSE":
        # Use Tensorflow
        import tensorflow as tf
        print(f"GPU available: {tf.test.is_gpu_available()}")
    else:
        # Use PyTorch
        import torch
        print(f"GPU available: {torch.cuda.is_available()}")

    # Prepare retriever
    doc_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name=args.model_name,
            database_path=f"./database/{args.model_name}",
        ),
    )
    doc_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl", batch_size=args.batch_size)
    doc_retrieval.add_corpus("tydiqa", "./corpus/tydiqa/corpus.jsonl", batch_size=args.batch_size)
    doc_retrieval.add_corpus("xquad", "./corpus/xquad/corpus.jsonl", batch_size=args.batch_size)
    doc_retrieval.add_corpus("miracl", "./corpus/miracl/corpus.jsonl", batch_size=args.batch_size)

    # Prepare benchmark
    benchmark = Benchmark(
        resource_management=ResourceManager(),
        retriever=doc_retrieval)
    benchmark.evaluate_on_datasets(
        ["iapp_wiki_qa", "tydiqa", "xquad", "miracl"]
    )
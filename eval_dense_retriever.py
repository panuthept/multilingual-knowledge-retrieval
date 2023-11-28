import argparse
from mkr.benchmark import RetrievalBenchmark
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mUSE")
    parser.add_argument("--model_checkpoint", type=str, default=None)
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
            model_checkpoint=args.model_checkpoint,
            database_path=f"./database/{args.model_name}/{args.model_checkpoint}",
        ),
    )
    # doc_retrieval.add_corpus("iapp_wiki_qa", "./datasets/thai_retrieval/iapp_wiki_qa/corpus.jsonl", batch_size=args.batch_size)
    # doc_retrieval.add_corpus("miracl", "./datasets/thai_retrieval/miracl/corpus.jsonl", batch_size=args.batch_size)
    doc_retrieval.add_corpus("thaiqa_squad", "./datasets/thai_retrieval/thaiqa_squad/corpus.jsonl", batch_size=args.batch_size)
    # doc_retrieval.add_corpus("tydiqa", "./datasets/thai_retrieval/tydiqa/corpus.jsonl", batch_size=args.batch_size)
    # doc_retrieval.add_corpus("xquad", "./datasets/thai_retrieval/xquad/corpus.jsonl", batch_size=args.batch_size)

    # Prepare benchmark
    benchmark = RetrievalBenchmark(
        resource_management=ResourceManager(),
        retriever=doc_retrieval)
    benchmark.evaluate_on_datasets(
        corpus_names=[
            # "iapp_wiki_qa", 
            # "miracl", 
            "thaiqa_squad", 
            # "tydiqa",
            # "xquad",
        ],
        dataset_names_or_paths=[
            # "./datasets/thai_retrieval/iapp_wiki_qa",
            # "./datasets/thai_retrieval/miracl",
            "./datasets/thai_retrieval/thaiqa_squad",
            # "./datasets/thai_retrieval/tydiqa",
            # "./datasets/thai_retrieval/xquad",
        ],
        split="test"
    )
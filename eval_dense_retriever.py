import argparse
from mkr.benchmark import RetrievalBenchmark
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
    doc_retrieval.add_corpus("th_xquad", "./corpus/xquad/th_corpus.jsonl", batch_size=args.batch_size)
    doc_retrieval.add_corpus("en_xquad", "./corpus/xquad/en_corpus.jsonl", batch_size=args.batch_size)

    # Prepare benchmark
    benchmark = RetrievalBenchmark(
        resource_management=ResourceManager(),
        retriever=doc_retrieval)
    benchmark.evaluate_on_datasets(
        dataset_names=["th_xquad", "en_xquad", "th_en_xquad", "en_th_xquad"],
        corpus_names=["th_xquad", "en_xquad", "en_xquad", "th_xquad"],
    )
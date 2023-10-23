import argparse
from mkr.benchmark import OpenedBookQABenchmark
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig
from mkr.question_answering.answer_extractor import AnswerExtractor, AnswerExtractorConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_name", type=str, default="mE5_small")
    parser.add_argument("--extractor_name", type=str, default="mRoBERTa")
    args = parser.parse_args()

    # Check GPU available
    # Use PyTorch
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")

    # Prepare models
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            model_name=args.retriever_name,
            database_path=f"./database/{args.retriever_name}",
        ),
    )
    extractor = AnswerExtractor(AnswerExtractorConfig(model_name=args.extractor_name))

    # Prepare benchmark
    benchmark = OpenedBookQABenchmark(
        resource_management=ResourceManager(),
        retriever=retriever,
        extractor=extractor
    )
    benchmark.evaluate_on_datasets(
        ["iapp_wiki_qa", "tydiqa", "xquad"]
    )
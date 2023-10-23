import argparse
from mkr.benchmark import ClosedBookQABenchmark
from mkr.resources.resource_manager import ResourceManager
from mkr.question_answering.answer_extractor import AnswerExtractor, AnswerExtractorConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mRoBERTa")
    args = parser.parse_args()

    # Check GPU available
    # Use PyTorch
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")

    # Prepare model
    extractor = AnswerExtractor(AnswerExtractorConfig(model_name=args.model_name))

    # Prepare benchmark
    benchmark = ClosedBookQABenchmark(
        resource_management=ResourceManager(),
        extractor=extractor)
    benchmark.evaluate_on_datasets(
        ["iapp_wiki_qa", "tydiqa", "xquad"]
    )
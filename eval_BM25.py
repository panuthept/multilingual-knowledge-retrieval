from mkr.benchmark import RetrievalBenchmark
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.sparse_retriever import SparseRetriever, SparseRetrieverConfig


if __name__ == "__main__":
    # Prepare retriever
    doc_retrieval = SparseRetriever(
        SparseRetrieverConfig(
            database_path="./database/BM25",
        ),
    )
    doc_retrieval.add_corpus("iapp_wiki_qa", "./datasets/thai_retrieval/iapp_wiki_qa/corpus.jsonl")
    doc_retrieval.add_corpus("miracl", "./datasets/thai_retrieval/miracl/corpus.jsonl")
    doc_retrieval.add_corpus("thaiqa_squad", "./datasets/thai_retrieval/thaiqa_squad/corpus.jsonl")
    doc_retrieval.add_corpus("tydiqa", "./datasets/thai_retrieval/tydiqa/corpus.jsonl")
    doc_retrieval.add_corpus("xquad", "./datasets/thai_retrieval/xquad/corpus.jsonl")

    # Prepare benchmark
    benchmark = RetrievalBenchmark(
        resource_management=ResourceManager(),
        retriever=doc_retrieval)
    benchmark.evaluate_on_datasets(
        corpus_names=[
            "iapp_wiki_qa", 
            "miracl", 
            "thaiqa_squad", 
            "tydiqa",
            "xquad",
        ],
        dataset_names_or_paths=[
            "./datasets/thai_retrieval/iapp_wiki_qa",
            "./datasets/thai_retrieval/miracl",
            "./datasets/thai_retrieval/thaiqa_squad",
            "./datasets/thai_retrieval/tydiqa",
            "./datasets/thai_retrieval/xquad",
        ],
        split="test"
    )
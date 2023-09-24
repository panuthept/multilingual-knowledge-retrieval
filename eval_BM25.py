from mkr.benchmark import Benchmark
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.sparse_retriever import SparseRetriever, SparseRetrieverConfig


if __name__ == "__main__":
    # Prepare retriever
    doc_retrieval = SparseRetriever(
        SparseRetrieverConfig(
            database_path="./database/BM25",
        ),
    )
    doc_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    doc_retrieval.add_corpus("tydiqa", "./corpus/tydiqa/corpus.jsonl")
    doc_retrieval.add_corpus("xquad", "./corpus/xquad/corpus.jsonl")
    doc_retrieval.add_corpus("miracl", "./corpus/miracl/corpus.jsonl")

    # Prepare benchmark
    benchmark = Benchmark(
        resource_management=ResourceManager(),
        retriever=doc_retrieval)
    benchmark.evaluate_on_datasets(
        ["iapp_wiki_qa", "tydiqa", "xquad", "miracl"]
    )
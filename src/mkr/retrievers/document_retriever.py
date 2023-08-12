import pandas as pd
from typing import List, Dict, Any
from mkr.utilities.general_utils import normalize_score
from mkr.resources.resource_manager import ResourceManager
from mkr.retrievers.baseclass import Retriever, RetrieverOutput


class DocumentRetriever(Retriever):
    def __init__(self, retriever: Retriever):
        self.resource_manager = ResourceManager(force_download=False)
        self.corpus = pd.read_csv(self.resource_manager.get_corpus_path("wikipedia_th_v2_raw"))
        self.retriever = retriever

    def __call__(self, queries: List[str], top_k: int = 3) -> RetrieverOutput:
        doc_resultss = []
        output = self.retriever(queries, top_k=top_k*10)
        for results in output.resultss:
            doc_results = {}
            for result in results.values():
                doc_id = result["doc_id"].split("-")[0]
                doc_title = result["doc_title"]
                content = self.corpus[self.corpus["title"] == doc_title]["text"].values[0]
                if doc_id not in doc_results:
                    doc_results[doc_id] = {
                        "doc_id": doc_id,
                        "score": result["score"],
                        "doc_url": result["doc_url"],
                        "doc_title": doc_title,
                        "doc_text": content,
                    }
                else:
                    doc_results[doc_id]["score"] += result["score"]
            doc_results = {doc_result["doc_id"]: doc_result for doc_result in sorted(doc_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]}
            doc_resultss.append(doc_results)
        # Normalize score
        doc_resultss = normalize_score(doc_resultss)
        return RetrieverOutput(
            queries=queries,
            resultss=doc_resultss,
        )
import pandas as pd
from typing import List, Dict
from mkr.retrievers.baseclass import Retriever
from mkr.resources.resource_manager import ResourceManager


class DocumentRetriever:
    def __init__(self, retriever: Retriever):
        self.resource_manager = ResourceManager(force_download=False)
        self.corpus = pd.read_csv(self.resource_manager.get_corpus_path("wikipedia_th_v2_raw"))
        self.retriever = retriever

    def __call__(self, queries: List[str], top_k: int = 3):
        doc_resultss = []
        resultss: List[List[Dict]] = self.retriever(queries, top_k=top_k*10)
        for results in resultss:
            doc_results = {}
            for result in results:
                doc_id = result["doc_id"].split("-")[0]
                doc_title = result["doc_title"]
                content = self.corpus[self.corpus["title"] == doc_title]["text"].values[0]
                if doc_id not in doc_results:
                    doc_results[doc_id] = {
                        "id": doc_id,
                        "score": result["score"],
                        "url": result["doc_url"],
                        "title": doc_title,
                        "content": content,
                    }
                else:
                    doc_results[doc_id]["score"] += result["score"]
            doc_results = sorted(doc_results.values(), key=lambda x: x["score"], reverse=True)
            doc_resultss.append(doc_results[:top_k])
        return doc_resultss
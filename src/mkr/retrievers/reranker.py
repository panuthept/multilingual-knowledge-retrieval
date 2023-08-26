from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
from mkr.models.mBERT import mBERTReranker
from mkr.databases.corpus_db import CorpusDB
from mkr.retrievers.baseclass import Retriever
from mkr.utilities.general_utils import read_corpus
from mkr.utilities.general_utils import normalize_score


@dataclass
class RerankerConfig:
    model_name: str
    database_path: str


class Reranker(Retriever):
    def __init__(self, config: RerankerConfig):
        self.model_name = config.model_name
        self.database_path = config.database_path

        self.model = self._load_model(self.model_name)
        self.corpus_db = CorpusDB(self.database_path)

    @staticmethod
    def _load_model(model_name: str):
        # Load encoder
        if model_name == "mBERT":
            model = mBERTReranker()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return model

    def add_corpus(self, corpus_name: str, corpus_path: str):
        if corpus_name in self.corpus_db.get_collection_names():
            return
        
        corpus_collection = self.corpus_db.create_or_get_collection(corpus_name)
        corpus = read_corpus(corpus_path)
        for doc in tqdm(corpus):
            # Add to database
            corpus_collection.add(
                ids=[doc["hash"]],
                contents=[doc["content"]],
                metadatas=[doc["metadata"]],
            )
        # Save database
        self.corpus_db.save()

    def __call__(
            self, 
            corpus_name: str, 
            query: str, 
            candidate_ids: List[str],
            top_k: int = 3, 
        ) -> List[Dict[str, Any]]:
        corpus_collection = self.corpus_db.create_or_get_collection(corpus_name)
        # Retrieve documents
        results = corpus_collection.retrieve(candidate_ids)
        # Scoring documents
        documents = [result["content"] for result in results]
        scores = self.model(query, documents)
        for result, score in zip(results, scores):
            result["score"] = score
        # Sorting results by score and get top-k results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        # Normalize score
        results = normalize_score(results)
        return results
    

if __name__ == "__main__":
    reranker = Reranker(
        RerankerConfig(
            model_name="mBERT", 
            database_path="./database/corpus_db"
        )
    )
    reranker.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    reranker.add_corpus("tydiqa_thai", "./corpus/tydiqa_thai/primary_corpus.jsonl")

    query = "คลีโอพัตราเป็นใคร"
    document_ids = [
        "8826153f3cadb1a1ea1edac969fd87ab1a39b48afd93827034e38e2574303ead",
        "7d8bdf7833ffcb65197ee2fe64ad6656e6701916c44ef675ad9c92c2ebf15646",
        "daa6f9ca03a753ce5bd338488628f876a9a9d4e405b7844aec96c75363acf446",
    ]
    reranker_results = reranker("iapp_wiki_qa", query, document_ids, top_k=3)
    print(reranker_results)
import os
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
from mkr.databases.bm25_db import BM25DB
from mkr.retrievers.baseclass import Retriever
from mkr.utilities.general_utils import read_corpus


@dataclass
class SparseRetrieverConfig:
    database_path: str


class SparseRetriever(Retriever):
    def __init__(self, config: SparseRetrieverConfig):
        self.database_path = config.database_path

        self.bm25_db = BM25DB(self.database_path)

    def add_corpus(self, corpus_name: str, corpus_path: str):
        if corpus_name in self.bm25_db.get_collection_names():
            return
        
        bm25_collection = self.bm25_db.create_or_get_collection(corpus_name)
        corpus = read_corpus(corpus_path)
        for doc in tqdm(corpus):
            bm25_collection.add(
                ids=[doc["hash"]],
                contents=[f"{doc['metadata']['title']}\n{doc['content']}"] if "title" in doc["metadata"] else [doc["content"]],
                metadatas=[doc["metadata"]],
            )
        bm25_collection.create_engine()
        # Save database
        self.bm25_db.save()

    def __call__(
            self, 
            corpus_name: str, 
            query: str, 
            top_k: int = 3, 
            candidate_ids: List[str] = None
        ) -> List[Dict[str, Any]]:
        bm25_collection = self.bm25_db.get_collection(corpus_name)
        # Retrieve documents
        results = bm25_collection.search(query, top_k=top_k, candidate_ids=candidate_ids)
        return results

    def save(self, path: str):
        # Save config
        config = {
            "database_path": self.database_path,
        }
        json.dump(config, open(os.path.join(path, "config.json"), "w"))

    @classmethod
    def from_config(cls, path):
        return cls(
            config=SparseRetrieverConfig(**json.load(open(os.path.join(path, "config.json"), "r")))
        )
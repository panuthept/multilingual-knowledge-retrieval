import os
import math
import json
from tqdm import trange
from typing import List
from dataclasses import dataclass
from mkr.vector_db.faiss_db import FaissVectorDB
from mkr.encoders.mUSE import mUSESentenceEncoder
from mkr.utilities.general_utils import read_corpus
from mkr.retrievers.baseclass import Retriever, RetrieverOutput


@dataclass
class DenseRetrieverConfig:
    model_name: str
    database_path: str


class DenseRetriever(Retriever):
    def __init__(self, config: DenseRetrieverConfig):
        self.model_name = config.model_name
        self.database_path = config.database_path

        self.encoder = self._load_encoder(self.model_name)
        self.vector_db = FaissVectorDB(self.database_path)

    def _load_encoder(self, model_name: str):
        # Load encoder
        if model_name == "mUSE":
            encoder = mUSESentenceEncoder()
        else:
            raise ValueError(f"Unknown encoder: {model_name}")
        return encoder
    
    def add_corpus(self, corpus_name: str, corpus_path: str, batch_size: int = 32):
        vector_collection = self.vector_db.get_collection(corpus_name)
        corpus = read_corpus(corpus_path)
        for batch_idx in trange(math.ceil(len(corpus) / batch_size)):
            batch_corpus = corpus[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            # Encode batch
            batch_ids = [doc["hash"] for doc in batch_corpus]
            batch_contents = [doc["content"] for doc in batch_corpus]
            batch_metadata = [doc["metadata"] for doc in batch_corpus]
            batch_embeddings = self.encoder.encode_batch(batch_contents, batch_size=batch_size)
            # Add to database
            vector_collection.add(
                ids=batch_ids,
                contents=batch_contents,
                vectors=batch_embeddings,
                metadatas=batch_metadata,
            )
        # Save database
        vector_collection.save()

    def __call__(self, corpus_name: str, queries: List[str], batch_size: int = 32, top_k: int = 3) -> RetrieverOutput:
        vector_collection = self.vector_db.get_collection(corpus_name)
        query_embeddings = self.encoder.encode_batch(queries, batch_size=batch_size)
        # Retrieve documents
        results = vector_collection.search(query_embeddings, top_k=top_k)
        return RetrieverOutput(
            queries=queries,
            results=results,
        )
    
    def save(self, path: str):
        # Save config
        config = {
            "model_name": self.model_name,
            "database_path": self.database_path,
        }
        json.dump(config, open(os.path.join(path, "config.json"), "w"))

    @classmethod
    def from_config(cls, path):
        return cls(
            config=DenseRetrieverConfig(**json.load(open(os.path.join(path, "config.json"), "r")))
        )
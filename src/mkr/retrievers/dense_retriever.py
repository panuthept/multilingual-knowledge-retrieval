import os
import math
import json
from tqdm import trange
from dataclasses import dataclass
from typing import List, Dict, Any
from mkr.databases.vector_db import VectorDB
from mkr.retrievers.baseclass import Retriever
from mkr.utilities.general_utils import read_corpus
from mkr.models.retrieval.mE5 import mE5SentenceEncoder
from mkr.models.retrieval.mUSE import mUSESentenceEncoder
from mkr.models.retrieval.mDPR import mDPRSentenceEncoder
from mkr.resources.resource_manager import ResourceManager
from mkr.models.retrieval.baseclass import SentenceEncoder
from mkr.resources.resource_constant import ENCODER_COLLECTION
from mkr.models.retrieval.mContriever import mContrieverSentenceEncoder


@dataclass
class DenseRetrieverConfig:
    model_name: str
    database_path: str
    model_checkpoint: str = None


class DenseRetriever(Retriever):
    def __init__(self, config: DenseRetrieverConfig):
        self.model_name = config.model_name
        self.model_checkpoint = config.model_checkpoint
        self.database_path = config.database_path

        self.resource_manager = ResourceManager()

        self.encoder: SentenceEncoder = self._load_encoder(self.model_name, self.model_checkpoint)
        self.vector_db = VectorDB(self.database_path)

    def _load_encoder(self, model_name: str, model_checkpoint: str = None) -> SentenceEncoder:
        # Load encoder
        # assert model_name in ENCODER_COLLECTION, f"Unknown encoder: {model_name}"
        model_checkpoint = model_checkpoint if model_checkpoint is not None else self.resource_manager.get_encoder_path(model_name)
        if model_name == "mUSE":
            encoder = mUSESentenceEncoder(model_checkpoint)
        elif model_name == "mE5":
            encoder = mE5SentenceEncoder(model_checkpoint)
        elif model_name == "mContriever":
            encoder = mContrieverSentenceEncoder(model_checkpoint)
        elif model_name == "mDPR":
            encoder = mDPRSentenceEncoder(model_checkpoint)
        else:
            raise ValueError(f"Unknown encoder: {model_name}")
        return encoder
    
    def add_corpus(self, corpus_name: str, corpus_path: str, batch_size: int = 32):
        if corpus_name in self.vector_db.get_collection_names():
            return
        
        vector_collection = self.vector_db.create_or_get_collection(corpus_name)
        corpus = read_corpus(corpus_path)
        for batch_idx in trange(math.ceil(len(corpus) / batch_size)):
            batch_corpus = corpus[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            # Encode batch
            batch_ids = [doc["id"] for doc in batch_corpus]
            batch_contents = [doc["content"] for doc in batch_corpus]
            batch_metadata = [None for doc in batch_corpus]
            # batch_metadata = [doc["metadata"] for doc in batch_corpus]
            # batch_titles = [metadata["title"] if "title" in metadata else None for metadata in batch_metadata]
            # batch_contents = [f"{title}\n{content}" if title is not None else content for title, content in zip(batch_titles, batch_contents)]
            batch_embeddings = self.encoder.encode_passages(batch_contents)
            # Add to database
            vector_collection.add(
                ids=batch_ids,
                contents=batch_contents,
                vectors=batch_embeddings,
                metadatas=batch_metadata,
            )
        # Save database
        self.vector_db.save()

    def __call__(
            self, 
            corpus_name: str, 
            query: str, 
            top_k: int = 3, 
            candidate_ids: List[str] = None
        ) -> List[Dict[str, Any]]:
        vector_collection = self.vector_db.create_or_get_collection(corpus_name)
        query_embedding = self.encoder.encode_queries(query)
        # Retrieve documents
        results = vector_collection.search(query_embedding, top_k=top_k, candidate_ids=candidate_ids)
        return results
    
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
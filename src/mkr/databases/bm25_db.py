import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi, BM25Plus, BM25L, BM25
from mkr.utilities.general_utils import normalize_score


class AutoBM25SeachEngine:
    @classmethod
    def create_engine(cls, corpus: List[str], tokenizer_name: str = "newmm", engine_name: str = "bm25_okapi"):
        # Tokenize corpus
        tokenized_corpus = [word_tokenize(text, engine=tokenizer_name) for text in corpus]
        # Create engine
        if engine_name == "bm25_okapi":
            engine = BM25Okapi(tokenized_corpus)
        elif engine_name == "bm25_plus":
            engine = BM25Plus(tokenized_corpus)
        elif engine_name == "bm25_l":
            engine = BM25L(tokenized_corpus)
        elif engine_name == "bm25":
            engine = BM25(tokenized_corpus)
        else:
            raise ValueError(f"Unknown BM25 model: {engine_name}")
        return engine


class BM25Collection:
    def __init__(self, collection_path: str, tokenizer_name: str = "newmm", engine_name: str = "bm25_okapi",):
        self.collection_path = collection_path
        self.tokenizer_name = tokenizer_name
        self.engine_name = engine_name
        # Initial parameters
        self.ids = []
        self.contents = []
        self.metadatas = []
        self.unique_ids = set()
        self.engine = None
        # Load parameters if exists
        if os.path.exists(self.collection_path):
            self.load()

    def add(
            self, 
            ids: List[str], 
            contents: List[str], 
            metadatas: List[Dict[str, Any]],
        ):
        # Add doc
        for content_id, content, metadata in zip(ids, contents, metadatas):
            if content_id in self.unique_ids:
                continue
            self.ids.append(content_id)
            self.contents.append(content)
            self.metadatas.append(metadata)
            self.unique_ids.add(content_id)

    def create_engine(self):
        self.engine = AutoBM25SeachEngine.create_engine(self.contents, self.tokenizer_name, self.engine_name)

    def search(
            self, 
            query: str, 
            top_k: int = 3, 
            candidate_ids: List[str] = None,
        ) -> List[Dict[str, Any]]:
        # Get search engine
        if self.engine is None:
            self.engine = AutoBM25SeachEngine.create_engine(self.contents, self.tokenizer_name, self.engine_name)
        # Search
        scores = self.engine.get_scores(word_tokenize(query, engine=self.tokenizer_name))
        if candidate_ids is not None:
            candidate_indices = [self.ids.index(candidate_id) for candidate_id in candidate_ids]
            scores = scores[candidate_indices]
        # Get top-k results
        topk_indices = np.argsort(scores)[::-1][:top_k]
        topk_scores = scores[topk_indices]

        results = []
        for score, index in zip(topk_scores, topk_indices):
            results.append({
                "id": self.ids[index],
                "content": self.contents[index],
                "metadata": self.metadatas[index],
                "score": score,
            })
        # Normalize scores
        results = normalize_score(results)
        return results
    
    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)

        if len(self.unique_ids) > 0:
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "w", encoding="utf-8") as f:
                for content_id, content, metadata in zip(self.ids, self.contents, self.metadatas):
                    f.write(json.dumps({
                        "id": content_id,
                        "content": content,
                        "metadata": metadata,
                    }, ensure_ascii=False))
                    f.write("\n")
            # Save engine
            pickle.dump(self.engine, open(os.path.join(self.collection_path, "engine.pkl"), "wb"))
            # Save config
            config = {
                "tokenizer_name": self.tokenizer_name,
                "engine_name": self.engine_name,
            }
            json.dump(config, open(os.path.join(self.collection_path, "config.json"), "w"))
        
    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.collection_path), f"Index directory not found: {self.collection_path}"

        if os.path.exists(os.path.join(self.collection_path, "engine.pkl")):
            self.ids = []
            self.contents = []
            self.metadatas = []
            self.unique_ids = set()
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "rb") as f:
                for line in f:
                    data = json.loads(line)
                    self.ids.append(data["id"])
                    self.contents.append(data["content"])
                    self.metadatas.append(data["metadata"])
                    self.unique_ids.add(data["id"])
            # Load engine
            self.engine = pickle.load(open(os.path.join(self.collection_path, "engine.pkl"), "rb"))
            # Load config
            config = json.load(open(os.path.join(self.collection_path, "config.json"), "r"))
            self.tokenizer_name = config["tokenizer_name"]
            self.engine_name = config["engine_name"]


class BM25DB:
    def __init__(self, database_path: str):
        self.database_path = database_path

        self.collections = {}       # {name: BM25Collection}
        self.collection_paths = {}  # {name: path}

        if os.path.exists(self.database_path):
            self.load()

    def get_collection_names(self) -> List[str]:
        return list(self.collection_paths.keys())

    def create_or_get_collection(self, name: str) -> BM25Collection:
        if name not in self.collection_paths:
            self.collection_paths[name] = os.path.join(self.database_path, name)
            self.collections[name] = BM25Collection(self.collection_paths[name])
        if name not in self.collections:
            self.collections[name] = BM25Collection(self.collection_paths[name])
        return self.collections[name]

    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        if len(self.collection_paths) > 0:
            # Save collection paths
            json.dump(self.collection_paths, open(os.path.join(self.database_path, "collection_paths.json"), "w"))
            # Save collections
            for collection in self.collections.values():
                collection.save()

    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.database_path), f"Index directory not found: {self.database_path}"

        if os.path.exists(os.path.join(self.database_path, "collection_paths.json")):
            # Load collection paths
            self.collection_paths = json.load(open(os.path.join(self.database_path, "collection_paths.json"), "r"))
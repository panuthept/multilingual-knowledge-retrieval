import os
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any


class VectorCollection:
    def __init__(self, collection_path: str, engine_name: str = "faiss",):
        self.collection_path = collection_path
        self.engine_name = engine_name
        # Initial parameters
        self.ids = []
        self.contents = []
        self.metadatas = []
        self.embeddings = None
        self.unique_ids = set()
        # Load parameters if exists
        if os.path.exists(self.collection_path):
            self.load()

    def _create_engine(self, embeddings: np.ndarray):
        if self.engine_name == "faiss":
            engine = faiss.IndexFlatIP(self.embeddings.shape[-1])
            engine.add(embeddings)
        return engine

    def add(
            self, 
            ids: List[str], 
            contents: List[str], 
            vectors: np.ndarray, 
            metadatas: List[Dict[str, Any]],
        ):
        # Initial embeddings
        if self.embeddings is None:
            self.embeddings = np.zeros((1000, vectors.shape[-1]))

        # Add doc to index
        prev_idx = len(self.unique_ids)
        new_indices = []
        for index, (content_id, content, metadata) in enumerate(zip(ids, contents, metadatas)):
            if content_id in self.unique_ids:
                continue
            new_indices.append(index)
            self.ids.append(content_id)
            self.contents.append(content)
            self.metadatas.append(metadata)
            self.unique_ids.add(content_id)
        end_index = prev_idx + len(new_indices)

        # Add embeddings
        # If embeddings is full, resize it
        if end_index > self.embeddings.shape[0]:
            new_size = self.embeddings.shape[0] * 2
            self.embeddings.resize(new_size, self.embeddings.shape[-1])
        # Add new embeddings
        self.embeddings[prev_idx:end_index] = vectors[new_indices]

    def search(
            self, 
            query_vector: np.ndarray, 
            top_k: int = 3, 
            candidate_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        # Get search engine
        if candidate_ids is not None:
            # If candidate_ids is provided, search only in candidate_ids
            # To do so, we need to create a new search engine
            candidate_indices = [self.ids.index(candidate_id) for candidate_id in candidate_ids]
            engine = self._create_engine(self.embeddings[candidate_indices])
        else:
            candidate_indices = np.arange(len(self.unique_ids))
            engine = self._create_engine(self.embeddings)

        # Search
        lst_scores, lst_indices = engine.search(query_vector, k=top_k)
        results = []
        for score, cand_index in zip(lst_scores[0], lst_indices[0]):
            # Filter indices that exceed the number of documents
            if cand_index >= len(candidate_indices):
                continue
            real_index = candidate_indices[cand_index]
            results.append({
                "id": self.ids[real_index],
                "content": self.contents[real_index],
                "metadata": self.metadatas[real_index],
                "score": score,
            })
            
        # Normalize scores
        max_score = max([result["score"] for result in results])
        for result in results:
            result["score"] = result["score"] / max_score
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
            # Save embeddings
            pickle.dump(self.embeddings, open(os.path.join(self.collection_path, "embeddings.pkl"), "wb"))
        
    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.collection_path), f"Index directory not found: {self.collection_path}"

        if os.path.exists(os.path.join(self.collection_path, "embeddings.pkl")):
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
            # Load embeddings
            self.embeddings = pickle.load(open(os.path.join(self.collection_path, "embeddings.pkl"), "rb"))


class VectorDB:
    def __init__(self, database_path: str):
        self.database_path = database_path

        self.collections = {}       # {name: VectorCollection}
        self.collection_paths = {}  # {name: path}

        if os.path.exists(self.database_path):
            self.load()

    def get_collection_names(self) -> List[str]:
        return list(self.collections.keys())

    def create_collection(self, name: str, force_create: bool = False):
        if name not in self.collections or force_create:
            self.collections[name] = VectorCollection(os.path.join(self.database_path, name))
            self.collection_paths[name] = os.path.join(self.database_path, name)

    def get_collection(self, name: str, force_create: bool = False) -> VectorCollection:
        if name not in self.collections or force_create:
            self.create_collection(name, force_create=force_create)
        return self.collections[name]

    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        if len(self.collections) > 0:
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
            # Load collections
            for name, path in self.collection_paths.items():
                self.collections[name] = VectorCollection(path)
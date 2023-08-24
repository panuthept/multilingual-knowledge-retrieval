import os
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any


class VectorCollection:
    def __init__(self, collection_path: str):
        self.collection_path = collection_path

        self.ids = []
        self.contents = []
        self.metadatas = []
        self.embeddings = None
        self.content_count = 0

        self.unique_ids = set()

        if os.path.exists(self.collection_path):
            self.load()
        else:
            os.makedirs(self.collection_path)

    def add(
            self, 
            ids: List[str], 
            contents: List[str], 
            vectors: np.ndarray, 
            metadatas: List[Dict[str, Any]],
        ):
        # Initial embeddings
        if self.embeddings is None:
            self.embeddings = np.zeros((0, vectors.shape[-1]))

        # Add doc to index
        new_indices = []
        for index, (content_id, content, metadata) in enumerate(zip(ids, contents, metadatas)):
            if content_id in self.unique_ids:
                continue
            new_indices.append(index)
            self.ids.append(content_id)
            self.contents.append(content)
            self.metadatas.append(metadata)
            # Update content_count
            self.content_count += 1
            self.unique_ids.add(content_id)
        self.embeddings = np.concatenate([self.embeddings, vectors[new_indices]], axis=0)

    def search(
            self, 
            query_vectors: np.ndarray, 
            top_k: int = 3, 
            engine_name: str = "faiss",
            candidate_ids: List[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        # Get search engine
        if engine_name == "faiss":
            search_engine = faiss.IndexFlatIP(self.embeddings.shape[-1])
            if candidate_ids is not None:
                candidate_indices = [self.ids.index(candidate_id) for candidate_id in candidate_ids]
            else:
                candidate_indices = np.arange(self.content_count)
            candidate_embeddings = self.embeddings[candidate_indices]
            search_engine.add(candidate_embeddings)
        # Search
        lst_scores, lst_indices = search_engine.search(query_vectors, k=top_k)
        lst_results = []
        for scores, indices in zip(lst_scores, lst_indices):
            results = []
            for score, cand_index in zip(scores, indices):
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
            lst_results.append(results)
        return lst_results
    
    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)

        if self.content_count > 0:
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

        if os.path.exists(os.path.join(self.collection_path, "content_count.pkl")):
            self.ids = []
            self.contents = []
            self.metadatas = []
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "rb") as f:
                for line in f:
                    data = json.loads(line)
                    self.ids.append(data["id"])
                    self.contents.append(data["content"])
                    self.metadatas.append(data["metadata"])
            self.content_count = len(self.ids)
            # Load embeddings
            self.embeddings = pickle.load(open(os.path.join(self.collection_path, "embeddings.pkl"), "rb"))


class VectorDB:
    def __init__(self, database_path: str):
        self.database_path = database_path

        self.collections = {}       # {name: VectorCollection}
        self.collection_paths = {}  # {name: path}

        if os.path.exists(self.database_path):
            self.load()
        else:
            os.makedirs(self.database_path)

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

        if os.path.exists(os.path.join(self.database_path, "collection_paths.pkl")):
            # Load collection paths
            self.collection_paths = json.load(open(os.path.join(self.database_path, "collection_paths.json"), "r"))
            # Load collections
            for name, path in self.collection_paths.items():
                self.collections[name] = VectorCollection(path)
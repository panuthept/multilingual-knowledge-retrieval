import os
import json
from typing import List, Dict, Any


class CorpusCollection:
    def __init__(self, collection_path: str):
        self.collection_path = collection_path
        # Initial parameters
        self.ids = []
        self.contents = []
        self.metadatas = []
        self.indexing = {}
        # Load parameters if exists
        if os.path.exists(self.collection_path):
            self.load()

    def add(
            self, 
            ids: List[str], 
            contents: List[str], 
            metadatas: List[Dict[str, Any]],
        ):
        # Add doc to index
        for content_id, content, metadata in zip(ids, contents, metadatas):
            if content_id in self.indexing:
                continue
            self.ids.append(content_id)
            self.contents.append(content)
            self.metadatas.append(metadata)
            self.indexing[content_id] = len(self.ids) - 1

    def retrieve(self, content_ids: List[str]) -> Dict[str, Any]:
        results = []
        for content_id in content_ids:
            if content_id not in self.indexing:
                return None
            index = self.indexing[content_id]
            results.append({
                "id": self.ids[index],
                "content": self.contents[index],
                "metadata": self.metadatas[index],
            })
        return results
    
    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)

        if len(self.indexing) > 0:
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "w", encoding="utf-8") as f:
                for content_id, content, metadata in zip(self.ids, self.contents, self.metadatas):
                    f.write(json.dumps({
                        "id": content_id,
                        "content": content,
                        "metadata": metadata,
                    }, ensure_ascii=False))
                    f.write("\n")
            # Save indexing as json
            json.dump(self.indexing, open(os.path.join(self.collection_path, "indexing.json"), "w"))
        
    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.collection_path), f"Index directory not found: {self.collection_path}"

        if os.path.exists(os.path.join(self.collection_path, "indexing.json")):
            self.ids = []
            self.contents = []
            self.metadatas = []
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "rb") as f:
                for line in f:
                    data = json.loads(line)
                    self.ids.append(data["id"])
                    self.contents.append(data["content"])
                    self.metadatas.append(data["metadata"])
            # Load indexing
            self.indexing = json.load(open(os.path.join(self.collection_path, "indexing.json"), "r"))


class CorpusDB:
    def __init__(self, database_path: str):
        self.database_path = database_path

        self.collections = {}       # {name: VectorCollection}
        self.collection_paths = {}  # {name: path}

        if os.path.exists(self.database_path):
            self.load()

    def get_collection_names(self) -> List[str]:
        return list(self.collection_paths.keys())

    def create_or_get_collection(self, name: str) -> CorpusCollection:
        if name not in self.collection_paths:
            self.collection_paths[name] = os.path.join(self.database_path, name)
            self.collections[name] = CorpusCollection(self.collection_paths[name])
        if name not in self.collections:
            self.collections[name] = CorpusCollection(self.collection_paths[name])
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
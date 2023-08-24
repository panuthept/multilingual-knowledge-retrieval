from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class RetrieverOutput:
    query: str
    results: List[Dict[str, Any]]


class Retriever(ABC):
    @abstractmethod
    def __call__(self, query: str, top_k: int = 3) :
        NotImplementedError
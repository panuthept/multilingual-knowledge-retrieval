from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class RetrieverOutput:
    queries: List[str]
    resultss: List[Dict[str, Dict[str, Any]]]


class Retriever(ABC):
    @abstractmethod
    def __call__(self, queries: List[str], top_k: int = 3) :
        NotImplementedError
from typing import List
from abc import ABC, abstractmethod



class Retriever(ABC):
    @abstractmethod
    def __call__(self, queries: List[str], top_k: int = 3):
        NotImplementedError
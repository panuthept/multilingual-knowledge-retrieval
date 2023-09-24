from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def __call__(self, query: str, top_k: int = 3):
        NotImplementedError
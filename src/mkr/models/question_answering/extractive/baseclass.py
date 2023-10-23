from typing import List
from abc import ABC, abstractmethod


class ExtractiveQA(ABC):
    @abstractmethod
    def predict_spans(self, queries: List[str], contexts: List[str]):
        raise NotImplementedError
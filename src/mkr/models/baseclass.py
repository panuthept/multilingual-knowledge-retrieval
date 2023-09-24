from typing import List, Union
from abc import ABC, abstractmethod


class SentenceEncoder(ABC):
    @abstractmethod
    def _encode_queries(self, queries: List[str]):
        raise NotImplementedError
    
    @abstractmethod
    def _encode_passages(self, passages: List[str]):
        raise NotImplementedError

    def encode_queries(self, queries: Union[List[str], str], return_numpy: bool = True):
        if isinstance(queries, str):
            queries = [queries]
        # Encode queries
        embedings = self._encode_queries(queries)
        # Cast to numpy
        if return_numpy:
            embedings = embedings.detach().numpy()
        return embedings
    
    def encode_passages(self, passages: Union[List[str], str], return_numpy: bool = True):
        if isinstance(passages, str):
            passages = [passages]
        # Encode passages
        embedings = self._encode_passages(passages)
        # Cast to numpy
        if return_numpy:
            embedings = embedings.detach().numpy()
        return embedings
        
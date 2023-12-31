import torch
from typing import List
from torch import Tensor
from torch.functional import F
from transformers import AutoTokenizer, AutoModel
from mkr.models.retrieval.baseclass import SentenceEncoder
from mkr.resources.resource_manager import ResourceManager


class mE5SentenceEncoder(SentenceEncoder):
    def __init__(self, model_checkpoint: str):
        self.resource_manager = ResourceManager()
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.model.eval()

        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _average_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _encode(self, texts: List[str]) -> Tensor:
        inputs = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        # Use GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = self._average_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Move to CPU if needed
        if torch.cuda.is_available():
            embeddings = embeddings.cpu()
        return embeddings
    
    def _encode_queries(self, queries: List[str]) -> Tensor:
        queries = [f"query: {query}" for query in queries]
        return self._encode(queries).detach()
    
    def _encode_passages(self, passages: List[str]) -> Tensor:
        passages = [f"passage: {passage}" for passage in passages]
        return self._encode(passages).detach()
    

# if __name__ == "__main__":
#     encoder = mE5SentenceEncoder("mE5_small")

#     english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
#     en_emb = encoder.encode_queries(english_sentences)
#     print(en_emb.shape)
#     en_emb = encoder.encode_passages(english_sentences)
#     print(en_emb.shape)
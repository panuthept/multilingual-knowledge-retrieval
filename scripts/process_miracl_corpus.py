import os
import json
from tqdm import tqdm
from hashlib import sha256
from datasets import load_dataset
from collections import defaultdict


corpus_name = "miracl"


def get_data(split="train", index=None):
    dataset = load_dataset("miracl/miracl", "th", split=split)
    
    qrels = defaultdict(set)
    index = {} if index is None else index
    progress_bar = tqdm(total=len(dataset))
    for sample in dataset:
        # Update progress bar
        progress_bar.update(1)
        # Extract query
        query = sample["query"]
        # Extract contexts
        # (Positive)
        positive_passages = sample["positive_passages"]
        for positive_passage in positive_passages:
            title = positive_passage["title"]
            content = positive_passage["text"]
            content_hash = sha256(content.encode('utf-8')).hexdigest()
            if content_hash not in index:
                # Add document to corpus
                with open(f"./corpus/{corpus_name}/corpus.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "hash": content_hash,
                        "content": content,
                        "metadata": {"title": title},
                    }, ensure_ascii=False))
                    f.write("\n")
                index[content_hash] = len(index)
            # Update qrels
            context_identifier = content_hash
            # Add query to qrels
            if query not in qrels:
                qrels[query] = set([context_identifier])
            else:
                if context_identifier not in qrels[query]:
                    # Add new context-answer pair
                    qrels[query].add(context_identifier)
        # (Negative)
        negative_passages = sample["negative_passages"]
        for negative_passage in negative_passages:
            title = negative_passage["title"]
            content = negative_passage["text"]
            content_hash = sha256(content.encode('utf-8')).hexdigest()
            if content_hash not in index:
                # Add document to corpus
                with open(f"./corpus/{corpus_name}/corpus.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "hash": content_hash,
                        "content": content,
                        "metadata": {"title": title},
                    }, ensure_ascii=False))
                    f.write("\n")
                index[content_hash] = len(index)

    # Convert set to list for answers in qrels
    lst_qrels = []
    for question, contexts in qrels.items():
        lst_qrels.append({
            "question": question,
            "contexts": list(contexts)
        })
    return index, lst_qrels


if __name__ == "__main__":
    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    # else:
    #     if os.path.exists(f"./corpus/{corpus_name}/corpus.jsonl"):
    #         raise Exception("Corpus already exists!")
        
    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    # else:
    #     if os.path.exists(f"./corpus/{corpus_name}/qrel_test.jsonl"):
    #         raise Exception("Qrels already exists!")
        
    index, qrels_train = get_data(split="train")
    index, qrels_dev = get_data(split="dev", index=index)

    json.dump(index, open(f"./corpus/{corpus_name}/corpus_index.json", "w", encoding="utf-8"))
    # Save qrels as a jsonl file
    with open(f"./corpus/{corpus_name}/qrel_train.jsonl", "w", encoding="utf-8") as f:
        for qrel in qrels_train:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open(f"./corpus/{corpus_name}/qrel_dev.jsonl", "w", encoding="utf-8") as f:
        for qrel in qrels_dev:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
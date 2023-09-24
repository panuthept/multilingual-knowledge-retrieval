import os
import json
from tqdm import tqdm
from hashlib import sha256
from datasets import load_dataset


corpus_name = "xquad"


def get_data():
    dataset = load_dataset("xquad", "xquad.th", split="validation")
    
    qrels = {}
    index = {}
    progress_bar = tqdm(total=len(dataset))
    for sample in dataset:
        # Update progress bar
        progress_bar.update(1)
        # Extract documents
        content = sample["context"]
        content_hash = sha256(content.encode('utf-8')).hexdigest()
        # Skip duplicate documents
        if content_hash not in index:
            # Add document to corpus
            with open(f"./corpus/{corpus_name}/corpus.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "hash": content_hash,
                    "content": content,
                    "metadata": {},
                }, ensure_ascii=False))
                f.write("\n")
            index[content_hash] = len(index)
        # Extract questions
        question = sample["question"]
        answers = [(answer_text, answer_start) for answer_text, answer_start in zip(sample["answers"]["text"], sample["answers"]["answer_start"])]
        context_identifier = content_hash
        # Add question to qrels
        if question not in qrels:
            qrels[question] = {context_identifier: set(answers)}
        else:
            if context_identifier not in qrels[question]:
                # Add new context-answer pair
                qrels[question][context_identifier] = set(answers)
            else:
                # Append answers to existing context-answer pair
                qrels[question][context_identifier].update(answers)

    # Convert set to list for answers in qrels
    lst_qrels = []
    for question, context_answers in qrels.items():
        lst_qrels.append({
            "question": question,
            "context_answers": {context_identifier: list(answer) for context_identifier, answer in context_answers.items()}
        })
    return index, lst_qrels


if __name__ == "__main__":
    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    else:
        if os.path.exists(f"./corpus/{corpus_name}/corpus.jsonl"):
            raise Exception("Corpus already exists!")
        
    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    else:
        if os.path.exists(f"./corpus/{corpus_name}/qrel_test.jsonl"):
            raise Exception("Qrels already exists!")
        
    index, qrels = get_data()

    json.dump(index, open(f"./corpus/{corpus_name}/corpus_index.json", "w", encoding="utf-8"))
    # Save qrels as a jsonl file
    with open(f"./corpus/{corpus_name}/qrel_test.jsonl", "w", encoding="utf-8") as f:
        for qrel in qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
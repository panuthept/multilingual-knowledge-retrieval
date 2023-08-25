import os
import json
from tqdm import tqdm
from hashlib import sha256
from datasets import load_dataset


corpus_name = "tydiqa_thai"


def get_primary_data(split="train", index=None):
    primary_dataset = load_dataset("tydiqa", "primary_task", split=split)

    qrels = {}
    index = {} if index is None else index
    progress_bar = tqdm(total=len(primary_dataset))
    for sample in primary_dataset:
        # Update progress bar
        progress_bar.update(1)
        # Extract documents
        title = sample["document_title"]
        content = sample["document_plaintext"]
        language = sample["language"]
        if language != "thai":
            continue
        url = sample["document_url"]
        b_content = bytes(content, "utf-8")
        # Get sub-contents
        sub_contents = []
        sub_content_indices = sample["passage_answer_candidates"]
        for start_idx, end_idx in zip(
            sub_content_indices["plaintext_start_byte"], 
            sub_content_indices["plaintext_end_byte"]
        ):
            sub_content = b_content[start_idx:end_idx].decode("utf-8")
            sub_content_hash = sha256(sub_content.encode('utf-8')).hexdigest()
            sub_contents.append({"content_hash": sub_content_hash, "content": sub_content})
            # Skip duplicate documents
            if sub_content_hash not in index:
                # Add document to corpus
                with open(f"./corpus/{corpus_name}/primary_corpus.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "hash": sub_content_hash,
                        "content": sub_content,
                        "metadata": {
                            "title": title,
                            "language": language,
                            "url": url,
                        },
                    }, ensure_ascii=False))
                    f.write("\n")
                index[sub_content_hash] = len(index)
        # Extract questions
        question = sample["question_text"]
        answer_indices = sample["annotations"]
        for sub_content_idx, start_idx, end_idx in zip(
            answer_indices["passage_answer_candidate_index"],
            answer_indices["minimal_answers_start_byte"],
            answer_indices["minimal_answers_end_byte"],
        ):
            if sub_content_idx == -1:
                if question not in qrels:
                    qrels[question] = {}
                continue
            context_identifier = f"{sub_contents[sub_content_idx]['content_hash']}"

            if start_idx == -1 or end_idx == -1:
                if question not in qrels:
                    qrels[question] = {context_identifier: set()}
                else:
                    if context_identifier not in qrels[question]:
                        # Add new context-answer pair
                        qrels[question][context_identifier] = set()
                continue
            answer = (b_content[start_idx:end_idx].decode("utf-8", errors='ignore'), start_idx)

            # Add question to qrels
            if question not in qrels:
                qrels[question] = {context_identifier: set([answer])}
            else:
                if context_identifier not in qrels[question]:
                    # Add new context-answer pair
                    qrels[question][context_identifier] = set([answer])
                else:
                    # Append answers to existing context-answer pair
                    qrels[question][context_identifier].add(answer)

    # Convert set to list for answers in qrels
    lst_qrels = []
    for question, context_answers in qrels.items():
        lst_qrels.append({
            "question": question,
            "context_answers": {context_identifier: list(answer) for context_identifier, answer in context_answers.items()}
        })
    return index, lst_qrels


def get_secondary_data(split="train", index=None):
    secondary_dataset = load_dataset("tydiqa", "secondary_task", split=split)

    qrels = {}
    index = {} if index is None else index
    progress_bar = tqdm(total=len(secondary_dataset))
    for sample in secondary_dataset:
        # Update progress bar
        progress_bar.update(1)
        # Extract documents
        title = sample["title"]
        content = sample["context"]
        content_hash = sha256(content.encode('utf-8')).hexdigest()
        # Skip duplicate documents
        if content_hash not in index:
            # Add document to corpus
            with open(f"./corpus/{corpus_name}/secondary_corpus.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "hash": content_hash,
                    "content": content,
                    "metadata": {
                        "title": title,
                    },
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
        if os.path.exists(f"./corpus/{corpus_name}/primary_corpus.jsonl"):
            raise Exception("Corpus already exists!")
        
    # Obtain corpus (primary task)
    primary_corpus_index, primary_qrel_train = get_primary_data(split="train")
    primary_corpus_index, primary_qrel_val = get_primary_data(split="validation", index=primary_corpus_index)

    json.dump(primary_corpus_index, open(f"./corpus/{corpus_name}/primary_corpus_index.json", "w", encoding="utf-8"))
    # Save qrels as a jsonl file
    with open(f"./corpus/{corpus_name}/primary_qrel_train.jsonl", "w", encoding="utf-8") as f:
        for qrel in primary_qrel_train:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open(f"./corpus/{corpus_name}/primary_qrel_val.jsonl", "w", encoding="utf-8") as f:
        for qrel in primary_qrel_val:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")


    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    else:
        if os.path.exists(f"./corpus/{corpus_name}/secondary_corpus.jsonl"):
            raise Exception("Corpus already exists!")
        
    # Obtain corpus (secondary task)
    secondary_corpus_index, secondary_qrel_train = get_secondary_data(split="train")
    secondary_corpus_index, secondary_qrel_val = get_secondary_data(split="validation", index=secondary_corpus_index)

    json.dump(secondary_corpus_index, open(f"./corpus/{corpus_name}/secondary_corpus_index.json", "w", encoding="utf-8"))
    # Save qrels as a jsonl file
    with open(f"./corpus/{corpus_name}/secondary_qrel_train.jsonl", "w", encoding="utf-8") as f:
        for qrel in secondary_qrel_train:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open(f"./corpus/{corpus_name}/secondary_qrel_val.jsonl", "w", encoding="utf-8") as f:
        for qrel in secondary_qrel_val:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
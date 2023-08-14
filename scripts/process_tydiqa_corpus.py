import os
import json
from tqdm import tqdm
from hashlib import sha256
from datasets import load_dataset
from collections import defaultdict


def get_primary_data(split="train"):
    primary_dataset = load_dataset("tydiqa", "primary_task", split=split)

    qrels = {}
    corpus = defaultdict(dict)
    progress_bar = tqdm(total=len(primary_dataset))
    for sample in primary_dataset:
        # Update progress bar
        progress_bar.update(1)
        # Extract documents
        title = sample["document_title"]
        content = sample["document_plaintext"]
        b_content = bytes(content, "utf-8")
        identifier = title
        # Get sub-contents
        sub_contents = []
        sub_content_indices = sample["passage_answer_candidates"]
        for start_idx, end_idx in zip(
            sub_content_indices["plaintext_start_byte"], 
            sub_content_indices["plaintext_end_byte"]
        ):
            sub_content = b_content[start_idx:end_idx].decode("utf-8")
            sub_identifier = sha256(sub_content.encode('utf-8')).hexdigest()
            sub_contents.append({"sub_identifier": sub_identifier, "content": sub_content})
            # Skip duplicate documents
            if identifier in corpus and sub_identifier in corpus[identifier]:
                continue
            # Add document to corpus
            corpus[identifier][sub_identifier] = {
                "content": sub_content,
                "metadata": {
                    "title": title,
                    "url": sample["document_url"],
                    "language": sample["language"],
                },
            }
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
            context_identifier = f"{identifier}-{sub_contents[sub_content_idx]['sub_identifier']}"

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
    return corpus, lst_qrels


def get_secondary_data(split="train"):
    secondary_dataset = load_dataset("tydiqa", "secondary_task", split=split)

    qrels = {}
    corpus = defaultdict(dict)
    progress_bar = tqdm(total=len(secondary_dataset))
    for sample in secondary_dataset:
        # Update progress bar
        progress_bar.update(1)
        # Extract documents
        title = sample["title"]
        content = sample["context"]
        identifier = title
        sub_identifier = sha256(content.encode('utf-8')).hexdigest()
        # Skip duplicate documents
        if identifier in corpus and sub_identifier in corpus[identifier]:
            continue
        # Add document to corpus
        corpus[identifier][sub_identifier] = {
            "content": content,
            "metadata": {
                "title": title,
                "language": sample["id"].split("-")[0],
            },
        }
        # Extract questions
        question = sample["question"]
        answers = [(answer_text, answer_start) for answer_text, answer_start in zip(sample["answers"]["text"], sample["answers"]["answer_start"])]
        context_identifier = f"{identifier}-{sub_identifier}"
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
    return corpus, lst_qrels


if __name__ == "__main__":
    # Obtain corpus (primary task)
    primary_corpus_train, primary_qrel_train = get_primary_data(split="train")
    primary_corpus_val, primary_qrel_val = get_primary_data(split="validation")

    progress_bar = tqdm(total=len(primary_corpus_val))
    for identifier in primary_corpus_val.keys():
        # Update progress bar
        progress_bar.update(1)
        for sub_identifier in primary_corpus_val[identifier].keys():
            if sub_identifier not in primary_corpus_train[identifier]:
                primary_corpus_train[identifier][sub_identifier] = primary_corpus_val[identifier][sub_identifier]

    if not os.path.exists("./corpus/tydiqa"):
        os.makedirs("./corpus/tydiqa")
    # Save corpus as a json file
    json.dump(primary_corpus_train, open("./corpus/tydiqa/primary_corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    # Save qrels as a jsonl file
    with open("./corpus/tydiqa/primary_qrel_train.jsonl", "w", encoding="utf-8") as f:
        for qrel in primary_qrel_train:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open("./corpus/tydiqa/primary_qrel_val.jsonl", "w", encoding="utf-8") as f:
        for qrel in primary_qrel_val:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")


    # Obtain corpus (secondary task)
    secondary_corpus_train, secondary_qrel_train = get_secondary_data(split="train")
    secondary_corpus_val, secondary_qrel_val = get_secondary_data(split="validation")

    # Merge corpus
    progress_bar = tqdm(total=len(secondary_corpus_val))
    for identifier in secondary_corpus_val.keys():
        # Update progress bar
        progress_bar.update(1)
        for sub_identifier in secondary_corpus_val[identifier].keys():
            if sub_identifier not in secondary_corpus_train[identifier]:
                secondary_corpus_train[identifier][sub_identifier] = secondary_corpus_val[identifier][sub_identifier]

    if not os.path.exists("./corpus/tydiqa"):
        os.makedirs("./corpus/tydiqa")
    # Save corpus as a json file
    json.dump(secondary_corpus_train, open("./corpus/tydiqa/secondary_corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    # Save qrels as a jsonl file
    with open("./corpus/tydiqa/secondary_qrel_train.jsonl", "w", encoding="utf-8") as f:
        for qrel in secondary_qrel_train:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open("./corpus/tydiqa/secondary_qrel_val.jsonl", "w", encoding="utf-8") as f:
        for qrel in secondary_qrel_val:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
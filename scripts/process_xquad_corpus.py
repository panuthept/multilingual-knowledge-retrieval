import os
import json
from tqdm import tqdm
from hashlib import sha256
from datasets import load_dataset


corpus_name = "xquad_original"


def get_data():
    th_dataset = load_dataset("xquad", "xquad.th", split="validation")
    en_dataset = load_dataset("xquad", "xquad.en", split="validation")
    assert len(th_dataset) == len(en_dataset)

    th_qrels = {}
    en_qrels = {}
    th_en_qrels = {}
    en_th_qrels = {}
    th_content_index = {}
    en_content_index = {}
    for th_sample, en_sample in tqdm(zip(th_dataset, en_dataset), total=len(th_dataset)):
        # Extract documents
        th_content = th_sample["context"]
        en_content = en_sample["context"]
        th_content_hash = sha256(th_content.encode('utf-8')).hexdigest()
        en_content_hash = sha256(en_content.encode('utf-8')).hexdigest()
        # Skip duplicate documents
        if th_content_hash not in th_content_index:
            # Save to disk
            with open(f"./corpus/{corpus_name}/th_corpus.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "hash": th_content_hash,
                    "content": th_content,
                    "metadata": {},
                }, ensure_ascii=False))
                f.write("\n")
            with open(f"./corpus/{corpus_name}/en_corpus.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "hash": en_content_hash,
                    "content": en_content,
                    "metadata": {},
                }, ensure_ascii=False))
                f.write("\n")
            th_content_index[th_content_hash] = len(th_content_index)
            en_content_index[en_content_hash] = len(en_content_index)
        # Extract questions
        th_question = th_sample["question"]
        en_question = en_sample["question"]
        th_answers = [(answer_text, answer_start) for answer_text, answer_start in zip(th_sample["answers"]["text"], th_sample["answers"]["answer_start"])]
        en_answers = [(answer_text, answer_start) for answer_text, answer_start in zip(en_sample["answers"]["text"], en_sample["answers"]["answer_start"])]
        # Add question to qrels
        # TH-TH
        if th_question not in th_qrels:
            th_qrels[th_question] = {th_content_hash: set(th_answers)}
        else:
            if th_content_hash not in th_qrels[th_question]:
                # Add new context-answer pair
                th_qrels[th_question][th_content_hash] = set(th_answers)
            else:
                # Append answers to existing context-answer pair
                th_qrels[th_question][th_content_hash].update(th_answers)
        # EN-EN
        if en_question not in en_qrels:
            en_qrels[en_question] = {en_content_hash: set(en_answers)}
        else:
            if en_content_hash not in en_qrels[en_question]:
                # Add new context-answer pair
                en_qrels[en_question][en_content_hash] = set(en_answers)
            else:
                # Append answers to existing context-answer pair
                en_qrels[en_question][en_content_hash].update(en_answers)
        # TH-EN
        if th_question not in th_en_qrels:
            th_en_qrels[th_question] = {en_content_hash: set(en_answers)}
        else:
            if en_content_hash not in th_en_qrels[th_question]:
                # Add new context-answer pair
                th_en_qrels[th_question][en_content_hash] = set(en_answers)
            else:
                # Append answers to existing context-answer pair
                th_en_qrels[th_question][en_content_hash].update(en_answers)
        # EN-TH
        if en_question not in en_th_qrels:
            en_th_qrels[en_question] = {th_content_hash: set(th_answers)}
        else:
            if th_content_hash not in en_th_qrels[en_question]:
                # Add new context-answer pair
                en_th_qrels[en_question][th_content_hash] = set(th_answers)
            else:
                # Append answers to existing context-answer pair
                en_th_qrels[en_question][th_content_hash].update(th_answers)
    # Convert qrels to list
    th_qrels_list = [
        {"question": question, "context_answers": {context_identifier: list(answer) for context_identifier, answer in context_answers.items()}} 
        for question, context_answers in th_qrels.items()
    ]
    en_qrels_list = [
        {"question": question, "context_answers": {context_identifier: list(answer) for context_identifier, answer in context_answers.items()}} 
        for question, context_answers in en_qrels.items()
    ]
    th_en_qrels_list = [
        {"question": question, "context_answers": {context_identifier: list(answer) for context_identifier, answer in context_answers.items()}} 
        for question, context_answers in th_en_qrels.items()
    ]
    en_th_qrels_list = [
        {"question": question, "context_answers": {context_identifier: list(answer) for context_identifier, answer in context_answers.items()}} 
        for question, context_answers in en_th_qrels.items()
    ]
    return th_content_index, en_content_index, th_qrels_list, en_qrels_list, th_en_qrels_list, en_th_qrels_list


if __name__ == "__main__":
    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    else:
        if os.path.exists(f"./corpus/{corpus_name}/th_corpus.jsonl"):
            raise Exception("TH Corpus already exists!")
        if os.path.exists(f"./corpus/{corpus_name}/en_corpus.jsonl"):
            raise Exception("EN Corpus already exists!")
        
    if not os.path.exists(f"./corpus/{corpus_name}"):
        os.makedirs(f"./corpus/{corpus_name}")
    else:
        if os.path.exists(f"./corpus/{corpus_name}/th_qrel_test.jsonl"):
            raise Exception("TH Qrels already exists!")
        if os.path.exists(f"./corpus/{corpus_name}/en_qrel_test.jsonl"):
            raise Exception("EN Qrels already exists!")
        if os.path.exists(f"./corpus/{corpus_name}/th_en_qrel_test.jsonl"):
            raise Exception("TH-EN Qrels already exists!")
        if os.path.exists(f"./corpus/{corpus_name}/en_th_qrel_test.jsonl"):
            raise Exception("EN-TH Qrels already exists!")
        
    th_corpus_index, en_corpus_index, th_qrels_list, en_qrels_list, th_en_qrels_list, en_th_qrels_list = get_data()

    json.dump(th_corpus_index, open(f"./corpus/{corpus_name}/th_corpus_index.json", "w", encoding="utf-8"))
    json.dump(en_corpus_index, open(f"./corpus/{corpus_name}/en_corpus_index.json", "w", encoding="utf-8"))
    # Save qrels as a jsonl file
    with open(f"./corpus/{corpus_name}/th_qrel_test.jsonl", "w", encoding="utf-8") as f:
        for qrel in th_qrels_list:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open(f"./corpus/{corpus_name}/en_qrel_test.jsonl", "w", encoding="utf-8") as f:
        for qrel in en_qrels_list:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open(f"./corpus/{corpus_name}/th_en_qrel_test.jsonl", "w", encoding="utf-8") as f:
        for qrel in th_en_qrels_list:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    with open(f"./corpus/{corpus_name}/en_th_qrel_test.jsonl", "w", encoding="utf-8") as f:
        for qrel in en_th_qrels_list:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
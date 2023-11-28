import os
import json
from tqdm import tqdm
from hashlib import sha256
from datasets import load_dataset
from collections import defaultdict


def load_tydiqa_primary_data(split="train", corpus_index=None):
    dataset = load_dataset("tydiqa", "primary_task", split=split)

    qrels = defaultdict(set)
    corpus_index = {} if corpus_index is None else corpus_index
    for sample in tqdm(dataset):
        if sample["language"] != "thai":
            continue
        document = sample["document_plaintext"]
        b_document = bytes(document, "utf-8")

        contexts = []
        for start_idx, end_idx in zip(
            sample["passage_answer_candidates"]["plaintext_start_byte"], 
            sample["passage_answer_candidates"]["plaintext_end_byte"]
        ):
            context = b_document[start_idx:end_idx].decode("utf-8")
            context_hash = sha256(context.encode('utf-8')).hexdigest()
            contexts.append({"hash": context_hash, "content": context})
            corpus_index[context_hash] = context
        
        question = sample["question_text"]
        for sub_content_idx, start_idx, end_idx in zip(
            sample["annotations"]["passage_answer_candidate_index"],
            sample["annotations"]["minimal_answers_start_byte"],
            sample["annotations"]["minimal_answers_end_byte"],
        ):
            if sub_content_idx == -1:
                continue
            context_hash = contexts[sub_content_idx]["hash"]
            qrels[question].add(context_hash)

    lst_qrels = [{"question": question, "context": list(context_hashes)} for question, context_hashes in qrels.items()]
    return corpus_index, lst_qrels


def load_iappwikiqa_data(corpus_index=None):
    dataset = json.load(open("./corpus/iapp_wiki_qa_original/iapp-thai-wikipedia-qa-1961-docs-9170-questions.json", "rb"))

    qrels = defaultdict(set)
    corpus_index = {} if corpus_index is None else corpus_index
    for sample in tqdm(dataset["db"].values()):
        if "title" not in sample or "detail" not in sample or "QA" not in sample:
            continue
        context = sample["detail"]
        context_hash = sha256(context.encode('utf-8')).hexdigest()
        corpus_index[context_hash] = context

        for qa in sample["QA"]:
            question = qa["q"]
            qrels[question].add(context_hash)

    lst_qrels = [{"question": question, "context": list(context_hashes)} for question, context_hashes in qrels.items()]
    return corpus_index, lst_qrels


def load_miracl_data(split="train", corpus_index=None):
    dataset = load_dataset("miracl/miracl", "th", split=split)

    qrels = defaultdict(set)
    corpus_index = {} if corpus_index is None else corpus_index
    for sample in tqdm(dataset):
        positive_contexts = sample["positive_passages"]
        negative_contexts = sample["negative_passages"]
        contexts = positive_contexts + negative_contexts
        for context in contexts:
            context_hash = sha256(context["text"].encode('utf-8')).hexdigest()
            corpus_index[context_hash] = context["text"]
        
        question = sample["query"]
        for positive_context in positive_contexts:
            context_hash = sha256(positive_context["text"].encode('utf-8')).hexdigest()
            qrels[question].add(context_hash)

    lst_qrels = [{"question": question, "context": list(context_hashes)} for question, context_hashes in qrels.items()]
    return corpus_index, lst_qrels


def load_xquad_data(corpus_index=None):
    dataset = load_dataset("xquad", "xquad.th", split="validation")

    qrels = defaultdict(set)
    corpus_index = {} if corpus_index is None else corpus_index
    for sample in tqdm(dataset):
        context = sample["context"]
        context_hash = sha256(context.encode('utf-8')).hexdigest()
        corpus_index[context_hash] = context

        question = sample["question"]
        qrels[question].add(context_hash)

    lst_qrels = [{"question": question, "context": list(context_hashes)} for question, context_hashes in qrels.items()]
    return corpus_index, lst_qrels


def load_thaiqa_squad_data(split="train", corpus_index=None):
    dataset = load_dataset("thaiqa_squad", split=split)

    qrels = defaultdict(set)
    corpus_index = {} if corpus_index is None else corpus_index
    for sample in tqdm(dataset):
        context = sample["context"]
        context = context.split("</doc>")[0]
        context = ">".join(context.split(">")[1:])

        context_hash = sha256(context.encode('utf-8')).hexdigest()
        corpus_index[context_hash] = context

        question = sample["question"]
        qrels[question].add(context_hash)

    lst_qrels = [{"question": question, "context": list(context_hashes)} for question, context_hashes in qrels.items()]
    return corpus_index, lst_qrels


def load_thaiqa_lst20_data(corpus_index=None):
    dataset = load_dataset("SuperAI2-Machima/ThaiQA_LST20", split="train")

    qrels = defaultdict(set)
    corpus_index = {} if corpus_index is None else corpus_index
    for sample in tqdm(dataset):
        if sample["status"] == 0:
            continue
        context = sample["context"]
        context_hash = sha256(context.encode('utf-8')).hexdigest()
        corpus_index[context_hash] = context

        question = sample["question"]
        qrels[question].add(context_hash)

    lst_qrels = [{"question": question, "context": list(context_hashes)} for question, context_hashes in qrels.items()]
    return corpus_index, lst_qrels


def save_dataset(name, save_path, corpus_index, train_qrels=None, dev_qrels=None, test_qrels=None):
    corpus_hash2id = {}
    for hash in corpus_index.keys():
        corpus_hash2id[hash] = len(corpus_hash2id)

    print(name)
    print(f"Corpus size: {len(corpus_index)}, Train: {len(train_qrels)}, Dev: {len(dev_qrels)}, Test: {len(test_qrels)}")

    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/corpus.json", "w", encoding="utf-8") as f:
        for hash, context in corpus_index.items():
            f.write(json.dumps({"id": corpus_hash2id[hash], "content": context}, ensure_ascii=False))
            f.write("\n")
    if train_qrels:
        train_qrels = [{"question": qrel["question"], "context_ids": [corpus_hash2id[context_hash] for context_hash in qrel["context"]]} for qrel in train_qrels]
        with open(f"{save_path}/train_qrels.jsonl", "w", encoding="utf-8") as f:
            for qrel in train_qrels:
                f.write(json.dumps(qrel, ensure_ascii=False))
                f.write("\n")
    if dev_qrels:
        dev_qrels = [{"question": qrel["question"], "context_ids": [corpus_hash2id[context_hash] for context_hash in qrel["context"]]} for qrel in dev_qrels]
        with open(f"{save_path}/dev_qrels.jsonl", "w", encoding="utf-8") as f:
            for qrel in dev_qrels:
                f.write(json.dumps(qrel, ensure_ascii=False))
                f.write("\n")
    if test_qrels:
        test_qrels = [{"question": qrel["question"], "context_ids": [corpus_hash2id[context_hash] for context_hash in qrel["context"]]} for qrel in test_qrels]
        with open(f"{save_path}/test_qrels.jsonl", "w", encoding="utf-8") as f:
            for qrel in test_qrels:
                f.write(json.dumps(qrel, ensure_ascii=False))
                f.write("\n")


if __name__ == "__main__":
    # Load corpus and qrels
    corpus_index, train_qrels = load_tydiqa_primary_data(split="train")
    corpus_index, test_qrels = load_tydiqa_primary_data(split="validation", corpus_index=corpus_index)
    train_qrels, dev_qrels = train_qrels[:int(len(train_qrels) * 0.9)], train_qrels[int(len(train_qrels) * 0.9):]
    save_dataset("tydiqa", "./datasets/thai_retrieval/tydiqa", corpus_index, train_qrels, dev_qrels, test_qrels)

    corpus_index, qrels = load_iappwikiqa_data()
    train_qrels = qrels[:int(len(qrels) * 0.8)]
    dev_qrels = qrels[int(len(qrels) * 0.8):int(len(qrels) * 0.9)]
    test_qrels = qrels[int(len(qrels) * 0.9):]
    save_dataset("iapp_wiki_qa", "./datasets/thai_retrieval/iapp_wiki_qa", corpus_index, train_qrels, dev_qrels, test_qrels)

    corpus_index, train_qrels = load_miracl_data(split="train")
    corpus_index, test_qrels = load_miracl_data(split="dev", corpus_index=corpus_index)
    train_qrels, dev_qrels = train_qrels[:int(len(train_qrels) * 0.9)], train_qrels[int(len(train_qrels) * 0.9):]
    save_dataset("miracl", "./datasets/thai_retrieval/miracl", corpus_index, train_qrels, dev_qrels, test_qrels)

    corpus_index, qrels = load_xquad_data()
    train_qrels = qrels[:int(len(qrels) * 0.8)]
    dev_qrels = qrels[int(len(qrels) * 0.8):int(len(qrels) * 0.9)]
    test_qrels = qrels[int(len(qrels) * 0.9):]
    save_dataset("xquad", "./datasets/thai_retrieval/xquad", corpus_index, train_qrels, dev_qrels, test_qrels)

    corpus_index, train_qrels = load_thaiqa_squad_data(split="train")
    corpus_index, test_qrels = load_thaiqa_squad_data(split="validation", corpus_index=corpus_index)
    train_qrels, dev_qrels = train_qrels[:int(len(train_qrels) * 0.9)], train_qrels[int(len(train_qrels) * 0.9):]
    save_dataset("thaiqa_squad", "./datasets/thai_retrieval/thaiqa_squad", corpus_index, train_qrels, dev_qrels, test_qrels)

    corpus_index, qrels = load_thaiqa_lst20_data()
    train_qrels = qrels[:int(len(qrels) * 0.8)]
    dev_qrels = qrels[int(len(qrels) * 0.8):int(len(qrels) * 0.9)]
    test_qrels = qrels[int(len(qrels) * 0.9):]
    save_dataset("thaiqa_lst20", "./datasets/thai_retrieval/thaiqa_lst20", corpus_index, train_qrels, dev_qrels, test_qrels)
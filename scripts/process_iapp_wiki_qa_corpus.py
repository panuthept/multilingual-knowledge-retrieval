import os
import json
from tqdm import tqdm
from hashlib import sha256
from collections import defaultdict


if __name__ == "__main__":
    dataset = json.load(open("./corpus/iapp_wiki_qa/iapp-thai-wikipedia-qa-1961-docs-9170-questions.json", "rb"))

    qrels = {}
    corpus = defaultdict(dict)
    progress_bar = tqdm(total=len(dataset["db"]))
    for sample in dataset["db"].values():
        # Update progress bar
        progress_bar.update(1)
        if "title" in sample and "detail" in sample and "QA" in sample:
            # Extract documents
            title = sample["title"]
            content = sample["detail"]
            identifier = title
            sub_identifier = sha256(content.encode('utf-8')).hexdigest()
            # Skip duplicate documents
            if identifier in corpus and sub_identifier in corpus[identifier]:
                continue
            # Add document to corpus
            corpus[identifier][sub_identifier] = {
                "content": content,
                "metadata": {"title": title},
            }
            # Extract answer spans
            answer_spans = {}
            for answer_span in sample["textDetector"]:
                answer_spans[answer_span["text"]] = answer_span["start"]
            # Extract questions
            for qa in sample["QA"]:
                question = qa["q"]
                answers = [(answer_text, answer_spans[answer_text]) for answer_text in qa["a"]]
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

    if not os.path.exists("./corpus/iapp_wiki_qa"):
        os.makedirs("./corpus/iapp_wiki_qa")
    # Save corpus as a json file
    json.dump(corpus, open("./corpus/iapp_wiki_qa/corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    # Save qrels as a jsonl file
    with open("./corpus/iapp_wiki_qa/qrels.jsonl", "w", encoding="utf-8") as f:
        for qrel in lst_qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
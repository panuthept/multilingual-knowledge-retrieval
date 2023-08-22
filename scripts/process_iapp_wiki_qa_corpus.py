import os
import json
from tqdm import tqdm
from hashlib import sha256


if __name__ == "__main__":
    dataset = json.load(open("./corpus/iapp_wiki_qa/iapp-thai-wikipedia-qa-1961-docs-9170-questions.json", "rb"))

    if not os.path.exists("./corpus/iapp_wiki_qa"):
        os.makedirs("./corpus/iapp_wiki_qa")
    else:
        if os.path.exists("./corpus/iapp_wiki_qa/corpus.jsonl"):
            raise Exception("Corpus already exists!")

    qrels = {}
    index = {}
    progress_bar = tqdm(total=len(dataset["db"]))
    for sample in dataset["db"].values():
        # Update progress bar
        progress_bar.update(1)
        if "title" in sample and "detail" in sample and "QA" in sample:
            # Extract documents
            title = sample["title"]
            content = sample["detail"]
            content_hash = sha256(content.encode('utf-8')).hexdigest()
            # Skip duplicate documents
            if content_hash not in index:
                # Add document to corpus
                with open(f"./corpus/iapp_wiki_qa/corpus.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "hash": content_hash,
                        "content": content,
                        "metadata": {
                            "title": title,
                        },
                    }, ensure_ascii=False))
                    f.write("\n")
                index[content_hash] = len(index)
            # Extract answer spans
            answer_spans = {}
            for answer_span in sample["textDetector"]:
                answer_spans[answer_span["text"]] = answer_span["start"]
            # Extract questions
            for qa in sample["QA"]:
                question = qa["q"]
                answers = [(answer_text, answer_spans[answer_text]) for answer_text in qa["a"]]
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

    json.dump(index, open("./corpus/iapp_wiki_qa/corpus_index.json", "w", encoding="utf-8"))
    # Save qrels as a jsonl file
    with open("./corpus/iapp_wiki_qa/qrels.jsonl", "w", encoding="utf-8") as f:
        for qrel in lst_qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
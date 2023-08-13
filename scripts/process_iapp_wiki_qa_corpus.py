import json


if __name__ == "__main__":
    dataset = json.load(open("./corpus/iapp_wiki_qa/iapp-thai-wikipedia-qa-1961-docs-9170-questions.json", "rb"))

    qrels = {}
    corpus = {}
    for sample in dataset["db"].values():
        if "title" in sample and "detail" in sample and "QA" in sample:
            # Extract documents
            identifier = sample["title"]
            # Skip duplicate documents
            if identifier in corpus:
                continue 
            # Add document to corpus
            title = sample["title"]
            content = sample["detail"]
            corpus[identifier] = {
                "content": content,
                "metadata": {"title": title},
            }
            # Extract questions
            for qa in sample["QA"]:
                question = qa["q"]
                answers = qa["a"]
                context = identifier
                # Add question to qrels
                if question not in qrels:
                    qrels[question] = {context: set(answers)}
                else:
                    if context not in qrels[question]:
                        # Add new context-answer pair
                        qrels[question][context] = set(answers)
                    else:
                        # Append answers to existing context-answer pair
                        qrels[question][context].update(answers)

    # Process qrels
    lst_qrels = []
    for question, context_answers in qrels.items():
        lst_qrels.append({
            "question": question,
            "context_answers": {context: list(answers) for context, answers in context_answers.items()}
        })

    # Save corpus as a json file
    json.dump(corpus, open("./corpus/iapp_wiki_qa/corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    # Save qrels as a jsonl file
    with open("./corpus/iapp_wiki_qa/qrels.jsonl", "w", encoding="utf-8") as f:
        for qrel in lst_qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
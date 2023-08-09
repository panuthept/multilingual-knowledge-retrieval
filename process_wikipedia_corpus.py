import json
import pandas as pd
from tqdm import trange

if __name__ == "__main__":
    df = pd.read_csv("./data/thaiwikipedia.csv")

    corpus = []
    for row in trange(len(df)):
        doc = df.iloc[row]
        doc_id = doc["id"]
        doc_url = doc["url"]
        doc_text = doc["text"]

        passage_count = 0
        for passage in doc_text.split("\n"):
            if len(passage) <= 40:
                continue
            corpus.append({"doc_id": f"{doc_id}-{passage_count}", "doc_url": doc_url, "doc_text": passage})
            passage_count += 1
    
    # Save to jsonl file
    with open("./data/wikipedia_th_corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
import json
import pandas as pd
from tqdm import trange

if __name__ == "__main__":
    df = pd.read_csv("./corpus/thaiwikipedia-new.csv")
    aux_df = pd.read_csv("./corpus/thaiwikipedia.csv")

    corpus = []
    for row in trange(len(df)):
        doc = df.iloc[row]
        doc_id = row
        doc_title = doc["title"]
        doc_text = doc["text"]
        if not isinstance(doc_text, str):
            continue
        doc_text = doc_text.strip().replace("\n\n", "\n")
        # Get document url
        doc_url = aux_df.where(aux_df["title"] == doc_title).dropna()["url"].values
        if len(doc_url) > 0:
            doc_url = doc_url[0]
        else:
            doc_url = None
        # Get passages in the document
        texts = []
        passage_count = 0
        for text in doc_text.split("\n"):
            if text == "" or text[0] == "|" or text[:2] == "{|" or text[:2] == "|}":
                continue
            if text == "== อ้างอิง ==" or text == "== แหล่งข้อมูลอื่น ==" or text == "==อ่านเพิ่ม==" or text == "== ดูเพิ่ม ==":
                break
            
            terminate_passage = False
            if text[:2] == "==":
                terminate_passage = True

            if terminate_passage:
                passage = "\n".join(texts)
                if len(passage) > 100:
                    corpus.append({
                        "doc_id": f"{doc_id}-{passage_count}", 
                        "doc_url": doc_url, 
                        "doc_title": doc_title, 
                        "doc_text": passage
                    })
                    passage_count += 1
                texts = []

            texts.append(text)

        passage = "\n".join(texts)
        if len(passage) > 100:
            corpus.append({
                "doc_id": f"{doc_id}-{passage_count}", 
                "doc_url": doc_url, 
                "doc_title": doc_title, 
                "doc_text": passage
            })
    
    # Save to jsonl file
    with open("./corpus/wikipedia_th_corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
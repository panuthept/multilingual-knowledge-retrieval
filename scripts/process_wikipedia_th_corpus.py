import os
import json
import pandas as pd
from tqdm import trange
from hashlib import sha256


def is_headline(text: str):
    return text[:2] == "==" and text[-2:] == "=="


def is_table(text: str):
    if text == "" or text[0] == "|" or text[:2] == "{|" or text[:2] == "|}":
        return True
    return False


def is_stop_headline(text: str):
    HEADLINES = ["อ้างอิง", "แหล่งข้อมูลอื่น", "อ่านเพิ่ม", "ดูเพิ่ม"]
    for headline in HEADLINES:
        if text == f"== {headline} ==" or text == f"=={headline}==":
            return True
    return False


if __name__ == "__main__":
    df = pd.read_csv("./corpus/wikipedia_th/thaiwikipedia_v2.csv")
    aux_df = pd.read_csv("./corpus/wikipedia_th/thaiwikipedia.csv")

    corpus_name = "corpus.jsonl"
    if not os.path.exists("./corpus/wikipedia_th"):
        os.makedirs("./corpus/wikipedia_th")
    else:
        if os.path.exists(f"./corpus/wikipedia_th/{corpus_name}"):
            raise Exception("Corpus already exists!")

    index = {}
    for row in trange(len(df)):
        doc = df.iloc[row]
        # Extract document content
        title = doc["title"]
        content = doc["text"]
        if not isinstance(content, str):
            continue
        content = content.strip().replace("\n\n", "\n")
        # Get document url
        urls = aux_df.where(aux_df["title"] == title).dropna()["url"].values
        if len(urls) > 0:
            url = urls[0]
        else:
            url = None
        # Extract sub-contents
        sub_content_fractions = []
        for sub_content_fraction in content.split("\n"):
            if sub_content_fraction == "" or is_table(sub_content_fraction):
                continue
            if is_stop_headline(sub_content_fraction):
                break
            
            if is_headline(sub_content_fraction):
                if len(sub_content_fractions) > 0 and not is_headline(sub_content_fractions[-1]):
                    sub_content = "\n".join(sub_content_fractions)
                    sub_content_hash = sha256(sub_content.encode('utf-8')).hexdigest()
                    if sub_content_hash not in index:
                        with open(f"./corpus/wikipedia_th/{corpus_name}", "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "hash": sub_content_hash,
                                "content": sub_content,
                                "metadata": {
                                    "title": title,
                                    "url": url,
                                },
                            }, ensure_ascii=False))
                            f.write("\n")
                        index[sub_content_hash] = len(index)
                sub_content_fractions = []
            sub_content_fractions.append(sub_content_fraction)
        
        if len(sub_content_fractions) > 0:
            if not is_headline(sub_content_fractions[-1]):
                sub_content = "\n".join(sub_content_fractions)
                sub_content_hash = sha256(sub_content.encode('utf-8')).hexdigest()
                if sub_content_hash not in index:
                    with open(f"./corpus/wikipedia_th/{corpus_name}", "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "hash": sub_content_hash,
                            "content": sub_content,
                            "metadata": {
                                "title": title,
                                "url": url,
                            },
                        }, ensure_ascii=False))
                        f.write("\n")
                    index[sub_content_hash] = len(index)

    json.dump(index, open("./corpus/wikipedia_th/corpus_index.json", "w", encoding="utf-8"))
import os
import json
import pandas as pd
from tqdm import trange
from hashlib import sha256
from collections import defaultdict


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

    corpus = defaultdict(dict)
    for row in trange(len(df)):
        doc = df.iloc[row]
        # Extract document content
        title = doc["title"]
        content = doc["text"]
        identifier = title
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
                sub_content = "\n".join(sub_content_fractions)
                sub_identifier = sha256(sub_content.encode('utf-8')).hexdigest()
                corpus[identifier][sub_identifier] = {
                    "content": sub_content,
                    "metadata": {
                        "title": title,
                        "url": url,
                    },
                }
            sub_content_fractions.append(sub_content_fraction)
        
        if len(sub_content_fractions) > 0:
            sub_content = "\n".join(sub_content_fractions)
            sub_identifier = sha256(sub_content.encode('utf-8')).hexdigest()
            corpus[identifier][sub_identifier] = {
                "content": sub_content,
                "metadata": {
                    "title": title,
                    "url": url,
                },
            }

    if not os.path.exists("./corpus/wikipedia_th"):
        os.makedirs("./corpus/wikipedia_th")
    # Save corpus as a json file
    json.dump(corpus, open("./corpus/wikipedia_th/wikipedia_th.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
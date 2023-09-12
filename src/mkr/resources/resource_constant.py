ENCODER_COLLECTION = {
    "mUSE": {
        "local_dir": "models",
        "file_name": "universal-sentence-encoder-multilingual_3",
        "download_url": "https://drive.google.com/uc?id=1MjDW6c2It-TM0YGoaYNh3Rq5P3cKvQRn",
        "download_output": "universal-sentence-encoder-multilingual_3.zip",
        "zip_file": True,
    },
    "finetuned_mUSE": {
        "local_dir": "models",
        "file_name": "finetuned_mUSE",
        "download_url": None,
        "download_output": None,
        "zip_file": True,
    }
}

CORPUS_COLLECTION = {
    "wikipedia_th": {
        "local_dir": "corpus/wikipedia_th",
        "file_name": "wikipedia_th.jsonl",
        "download_url": "https://drive.google.com/uc?id=1GtoAS1QCMLOxuE2zva8uGoF9U_XlASei",
        "download_output": "wikipedia_th.jsonl",
        "zip_file": False,
    },
    "wikipedia_th_v2": {
        "local_dir": "corpus/wikipedia_th",
        "file_name": "wikipedia_th_v2.jsonl",
        "download_url": "https://drive.google.com/uc?id=1F7EyG4pzvVLSC4bubbAN1c6sAQ00PZQK",
        "download_output": "wikipedia_th_v2.jsonl.zip",
        "zip_file": True,
    },
    "wikipedia_th_v2_raw": {
        "local_dir": "corpus/wikipedia_th",
        "file_name": "thaiwikipedia_v2.csv",
        "download_url": "https://drive.google.com/uc?id=1lkMQ189Xq1csyBJN9HmMJV3FMAwTjAb4",
        "download_output": "thaiwikipedia_v2.csv.zip",
        "zip_file": True,
    }
}

INDEX_COLLECTION = {
    "wikipedia_th_bm25_okapi_newmm": {
        "local_dir": "indexes",
        "file_name": "wikipedia_th_bm25_okapi_newmm",
        "download_url": "https://drive.google.com/uc?id=1y7wVOhvorKIBzbmXFhkzjq8PvLkZ5Xq5",
        "download_output": "wikipedia_th_bm25_okapi_newmm.zip",
        "zip_file": True,
    },
    "wikipedia_th_mUSE": {
        "local_dir": "indexes",
        "file_name": "wikipedia_th_mUSE",
        "download_url": "https://drive.google.com/uc?id=1KF63e784iEmuGycw1GFH6qi1aXwrsQSQ",
        "download_output": "wikipedia_th_mUSE.zip",
        "zip_file": True,
    },
    "wikipedia_th_v2_bm25_okapi_newmm": {
        "local_dir": "indexes",
        "file_name": "wikipedia_th_v2_bm25_okapi_newmm",
        "download_url": "https://drive.google.com/uc?id=1K7AKYyqktabUd8CLiVsSJ1yDqwwE81SE",
        "download_output": "wikipedia_th_v2_bm25_okapi_newmm.zip",
        "zip_file": True,
    },
    "wikipedia_th_v2_mUSE": {
        "local_dir": "indexes",
        "file_name": "wikipedia_th_v2_mUSE",
        "download_url": "https://drive.google.com/uc?id=1LqAUud4wUXnERXD3rjlpPCg-L7gs5Uja",
        "download_output": "wikipedia_th_v2_mUSE.zip",
        "zip_file": True,
    }
}
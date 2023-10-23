EXTRACTIVEQA_COLLECTION = {
    "mRoBERTa": {
        "local_dir": "models/alon-albalak/xlm-roberta-base-xquad",
        "hf_model_name": "alon-albalak/xlm-roberta-base-xquad",
        "hf_model_type": "question_answering",
    }
}

ENCODER_COLLECTION = {
    "mUSE": {
        "local_dir": "models/universal-sentence-encoder-multilingual_3",
        "file_name": "universal-sentence-encoder-multilingual_3",
        "download_url": "https://drive.google.com/uc?id=1MjDW6c2It-TM0YGoaYNh3Rq5P3cKvQRn",
        "download_output": "universal-sentence-encoder-multilingual_3.zip",
        "zip_file": True,
    },
    "finetuned_mUSE": {
        "local_dir": "models/finetuned_mUSE",
        "file_name": "finetuned_mUSE",
        "download_url": None,
        "download_output": None,
        "zip_file": True,
    },
    "mE5_base": {
        "local_dir": "models/multilingual-e5-base",
        "download_method": "huggingface",
        "download_url": "intfloat/multilingual-e5-base",
    },
    "mE5_small": {
        "local_dir": "models/multilingual-e5-small",
        "download_method": "huggingface",
        "download_url": "intfloat/multilingual-e5-small",
    },
    "mE5_large": {
        "local_dir": "models/multilingual-e5-large",
        "download_method": "huggingface",
        "download_url": "intfloat/multilingual-e5-large",
    },
    "mContriever": {
        "local_dir": "models/mcontriever",
        "download_method": "huggingface",
        "download_url": "facebook/mcontriever",
    },
    "mContriever_msmarco": {
        "local_dir": "models/mcontriever-msmarco",
        "download_method": "huggingface",
        "download_url": "facebook/mcontriever-msmarco",
    },
    "mDPR": {},
    "mDPR_query": {
        "local_dir": "models/mdpr_question_nq",
        "download_method": "huggingface",
        "download_url": "castorini/mdpr-question-nq",
    },
    "mDPR_passage": {
        "local_dir": "models/mdpr_passage_nq",
        "download_method": "huggingface",
        "download_url": "castorini/mdpr-passage-nq",
    },
    "mDPR_msmarco": {},
    "mDPR_query_msmarco": {
        "local_dir": "models/mdpr_question_msmarco",
        "download_method": "huggingface",
        "download_url": "crystina-z/mdpr-question-msmarco",
    },
    "mDPR_passage_msmarco": {
        "local_dir": "models/mdpr_passage_msmarco",
        "download_method": "huggingface",
        "download_url": "crystina-z/mdpr-passage-msmarco",
    },
    "mDPR_tied": {
        "local_dir": "models/mdpr_tied_nq",
        "download_method": "huggingface",
        "download_url": "castorini/mdpr-tied-pft-nq",
    },
    "mDPR_tied_msmarco": {
        "local_dir": "models/mdpr_tied_msmarco",
        "download_method": "huggingface",
        "download_url": "castorini/mdpr-tied-pft-msmarco",
    },
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
    },
    "iapp_wiki_qa": {
        "local_dir": "corpus/iapp_wiki_qa",
    },
    "tydiqa": {
        "local_dir": "corpus/tydiqa",
    },
    "xquad": {
        "local_dir": "corpus/xquad",
    },
    "miracl": {
        "local_dir": "corpus/miracl",
    },
}

RETRIEVAL_DATASET_COLLECTION = {
    "iapp_wiki_qa": {
        "local_dir": "datasets/retrieval/iapp_wiki_qa",
    },
    "tydiqa": {
        "local_dir": "datasets/retrieval/tydiqa",
    },
    "xquad": {
        "local_dir": "datasets/retrieval/xquad",
    },
    "miracl": {
        "local_dir": "datasets/retrieval/miracl",
    },
}

QA_DATASET_COLLECTION = {
    "iapp_wiki_qa": {
        "local_dir": "datasets/question_answering/iapp_wiki_qa",
    },
    "tydiqa": {
        "local_dir": "datasets/question_answering/tydiqa",
    },
    "xquad": {
        "local_dir": "datasets/question_answering/xquad",
    },
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
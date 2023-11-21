import os
import gdown
import shutil
from typing import Dict
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from mkr.resources.resource_constant import CORPUS_COLLECTION, RETRIEVAL_DATASET_COLLECTION, QA_DATASET_COLLECTION, ENCODER_COLLECTION, INDEX_COLLECTION, EXTRACTIVEQA_COLLECTION


class ResourceManager:
    def __init__(self, resource_dir: str = "./", force_download: bool = False):
        self.resource_dir = resource_dir
        self.force_download = force_download

    def download_hf_model(self, model_name: str, model_type: str, file_dir: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == "question_answering":
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        # Save tokenizer and model
        tokenizer.save_pretrained(file_dir)
        model.save_pretrained(file_dir)

    def download_resource_from_huggingface(self, resource_details: Dict[str, str], file_dir: str):
        tokenizer = AutoTokenizer.from_pretrained(resource_details["download_url"])
        model = AutoModel.from_pretrained(resource_details["download_url"])
        # Save tokenizer and model
        tokenizer.save_pretrained(file_dir)
        model.save_pretrained(file_dir)

    def download_resource_if_needed(self, resource_details: Dict[str, str], force_download: bool = False):
        force_download = force_download or self.force_download
        # Create base_dir if not exists
        local_dir = os.path.join(self.resource_dir, resource_details["local_dir"])
        base_dir = os.path.dirname(local_dir)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # Check if the resource is already downloaded
        if os.path.exists(local_dir) and not force_download:
            return
        
        # Download the resource
        if resource_details["download_method"] == "huggingface":
            self.download_resource_from_huggingface(resource_details, local_dir)
        else:
            download_url = resource_details["download_url"]
            download_output = os.path.join(base_dir, resource_details["download_output"])
            if not os.path.exists(download_output):
                gdown.download(download_url, download_output, quiet=False)
            # Process the downloaded file
            if resource_details["zip_file"]:
                shutil.unpack_archive(download_output, base_dir)
                os.remove(download_output)

    def get_corpus_path(self, corpus_name: str, download: bool = True, force_download: bool = False):
        # if download:
        #     self.download_resource_if_needed(CORPUS_COLLECTION[corpus_name], force_download=force_download)

        if corpus_name in CORPUS_COLLECTION:
            resource_details = CORPUS_COLLECTION[corpus_name]
            corpus_path = os.path.join(self.resource_dir, resource_details["local_dir"])
        else:
            raise ValueError(f"Unknown corpus: {corpus_name}")
        return corpus_path
    
    def get_dataset_path(self, dataset_name: str, dataset_type: str, download: bool = True, force_download: bool = False):
        if dataset_type == "question_answering":
            assert dataset_name in QA_DATASET_COLLECTION, f"Unknown dataset name: {dataset_name}"
            resource_details = QA_DATASET_COLLECTION[dataset_name]
        elif dataset_type == "retrieval":
            assert dataset_name in RETRIEVAL_DATASET_COLLECTION, f"Unknown dataset name: {dataset_name}"
            resource_details = RETRIEVAL_DATASET_COLLECTION[dataset_name]

        dataset_path = os.path.join(self.resource_dir, resource_details["local_dir"])
        return dataset_path
    
    def get_model_path(self, model_name: str, model_type: str, download: bool = True, force_download: bool = False):
        if model_type == "extractive_question_answering":
            assert model_name in EXTRACTIVEQA_COLLECTION, f"Unknown model name: {model_name}"
            resource_details = EXTRACTIVEQA_COLLECTION[model_name]
        elif model_type == "retrieval":
            assert model_name in ENCODER_COLLECTION, f"Unknown model name: {model_name}"
            resource_details = ENCODER_COLLECTION[model_name]

        if force_download or (download and not os.path.exists(resource_details["local_dir"])):
            self.download_hf_model(model_name=resource_details["hf_model_name"], model_type=resource_details["hf_model_type"], file_dir=resource_details["local_dir"])

        encoder_path = os.path.join(self.resource_dir, resource_details["local_dir"])
        return encoder_path

    def get_encoder_path(self, encoder_name: str, download: bool = True, force_download: bool = False):
        if download:
            self.download_resource_if_needed(ENCODER_COLLECTION[encoder_name], force_download=force_download)

        if encoder_name in ENCODER_COLLECTION:
            resource_details = ENCODER_COLLECTION[encoder_name]
            encoder_path = os.path.join(self.resource_dir, resource_details["local_dir"])
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        return encoder_path

    def get_index_path(self, index_name: str, download: bool = True, force_download: bool = False):
        if download:
            self.download_resource_if_needed(INDEX_COLLECTION[index_name], force_download=force_download)

        if index_name in INDEX_COLLECTION:
            resource_details = INDEX_COLLECTION[index_name]
            index_path = os.path.join(
                self.resource_dir, resource_details["local_dir"], resource_details["file_name"]
            )
        else:
            raise ValueError(f"Unknown index: {index_name}")
        return index_path
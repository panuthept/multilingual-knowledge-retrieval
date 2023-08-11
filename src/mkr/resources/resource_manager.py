import os
import gdown
import shutil
from typing import Dict
from mkr.resources.resource_constant import CORPUS_COLLECTION, ENCODER_COLLECTION, INDEX_COLLECTION


class ResourceManager:
    def __init__(self, resource_dir: str = "./", force_download: bool = False):
        self.resource_dir = resource_dir
        self.force_download = force_download

    def download_resource_if_needed(self, resource_details: Dict[str, str], force_download: bool = False):
        force_download = force_download or self.force_download
        # Create local_dir if not exists
        local_dir = os.path.join(self.resource_dir, resource_details["local_dir"])
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        # Check if the resource is already downloaded
        file_dir = os.path.join(local_dir, resource_details["file_name"])
        if os.path.exists(file_dir) and not force_download:
            return
        # Download the resource
        download_url = resource_details["download_url"]
        download_output = os.path.join(local_dir, resource_details["download_output"])
        if not os.path.exists(download_output):
            gdown.download(download_url, download_output, quiet=False)
        # Process the downloaded file
        if resource_details["zip_file"]:
            shutil.unpack_archive(download_output, local_dir)
            os.remove(download_output)

    def get_corpus_path(self, corpus_name: str, download: bool = True, force_download: bool = False):
        if download:
            self.download_resource_if_needed(CORPUS_COLLECTION[corpus_name], force_download=force_download)

        if corpus_name in CORPUS_COLLECTION:
            resource_details = CORPUS_COLLECTION[corpus_name]
            corpus_path = os.path.join(
                self.resource_dir, resource_details["local_dir"], resource_details["file_name"]
            )
        else:
            raise ValueError(f"Unknown corpus: {corpus_name}")
        return corpus_path

    def get_encoder_path(self, encoder_name: str, download: bool = True, force_download: bool = False):
        if download:
            self.download_resource_if_needed(ENCODER_COLLECTION[encoder_name], force_download=force_download)

        if encoder_name in ENCODER_COLLECTION:
            resource_details = ENCODER_COLLECTION[encoder_name]
            encoder_path = os.path.join(
                self.resource_dir, resource_details["local_dir"], resource_details["file_name"]
            )
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
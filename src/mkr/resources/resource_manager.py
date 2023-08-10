import os
import tensorflow_hub
from typing import Dict
from mkr.resources.resource_constant import ENCODER_COLLECTION


class ResourceManager:
    def __init__(self, resource_dir: str = "./resources"):
        self.resource_dir = resource_dir

    def download_resource_if_needed(self, resource_details: Dict[str, str]):
        # Create local_dir if not exists
        local_dir = os.path.join(self.resource_dir, resource_details["local_dir"])
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        # Check if the resource is already downloaded
        file_dir = os.path.join(local_dir, resource_details["file_name"])
        if os.path.exists(file_dir):
            return
        # Download the resource
        download_url = resource_details["download_url"]
        downloaded_file_dir = os.path.join(local_dir, resource_details["downloaded_file_name"])
        if not os.path.exists(downloaded_file_dir):
            os.system(f"wget {download_url} -O {downloaded_file_dir}")
        # Process the downloaded file
        file_type = resource_details["file_type"]
        if file_type == "tar.gz":
            os.makedirs(file_dir)
            os.system(f"tar -xvf {downloaded_file_dir} -C {file_dir}")
            os.system(f"rm {downloaded_file_dir}")

    def get_encoder(self, encoder_name: str):
        self.download_resource_if_needed(ENCODER_COLLECTION[encoder_name])
        if encoder_name == "mUSE":
            resource_details = ENCODER_COLLECTION[encoder_name]
            encoder = tensorflow_hub.load(os.path.join(
                self.resource_dir, resource_details["local_dir"], resource_details["file_name"]
            ))
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        return encoder
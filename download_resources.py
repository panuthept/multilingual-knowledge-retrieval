import argparse
from mkr.resources.resource_manager import ResourceManager


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    resource_manager = ResourceManager(force_download=args.force_download)
    resource_manager.get_corpus_path("wikipedia_th_v2_raw", download=True)
    resource_manager.get_corpus_path("wikipedia_th_v2", download=True)
    resource_manager.get_encoder_path("mUSE", download=True)
    resource_manager.get_index_path("wikipedia_th_v2_mUSE", download=True)
    resource_manager.get_index_path("wikipedia_th_v2_bm25_okapi_newmm", download=True)
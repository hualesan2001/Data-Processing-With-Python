import os
import urllib.request
import sys
import zipfile
import logging
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class FMADataset:
    """Download and unzip the FMA metadata + fma_small audio archives into ``./data/``. Idempotent."""

    URL_metadata: str = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    URL_dataset_small: str = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    data_dir: str = "./data/"

    @staticmethod
    def _download(url: str, folder:str) -> str:
        """Download ``url`` into ``folder`` if not already present; return the local file path."""
        if not os.path.isdir(folder):
            os.mkdir(folder)
        basename = os.path.basename(url)
        path = os.path.join(folder, basename)

        if not os.path.exists(path):
            logger.info(f"Downloading {basename}...")
            urllib.request.urlretrieve(url, filename=f"{path}.tmp")
            os.rename(f"{path}.tmp", path)
        return path

    @staticmethod
    def unzip(path:str) -> None:
        """Extract a zip archive next to itself; skip if the target folder already exists."""
        extract_dir = os.path.dirname(path)
        folder_name = os.path.basename(path).replace(".zip", "")
        unzipped_folder_path = os.path.join(extract_dir, folder_name)

        if os.path.isdir(unzipped_folder_path):
            logger.info(f"Skipping unzip, {unzipped_folder_path} already exists.")
            return

        logger.info(f"Unpacking {path}...")
        with zipfile.ZipFile(path, "r") as zip_file:
            members = zip_file.infolist()
            with tqdm(members, desc=f"Extracting {folder_name}", unit="file") as bar:
                for member in bar:
                    zip_file.extract(member, extract_dir)

    def __init__(self):
        """Fetch and unzip both the metadata and the fma_small audio archive."""
        logger.info(f"Downloading FMA datasets")
        metadata_path = FMADataset._download(self.URL_metadata, self.data_dir)
        dataset_path = FMADataset._download(self.URL_dataset_small, self.data_dir)
        logger.info(f"Unpacking FMA datasets")
        FMADataset.unzip(metadata_path)
        FMADataset.unzip(dataset_path)
        logger.info(f"Finished unpacking FMA datasets")


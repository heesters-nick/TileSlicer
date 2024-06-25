import logging
import os
import subprocess
import sys

import vos
from tqdm import tqdm

sys.path.append('/home/heesters/projects/def-sfabbro/heesters/github/TileSlicer')
from data_utils import setup_logging


def list_vospace(directory):
    """Use vls to list contents of a VOSpace directory"""
    try:
        result = subprocess.run(['vls', directory], text=True, capture_output=True, check=True)
        entry_list = [line.strip() for line in result.stdout.splitlines()]
        return entry_list
    except subprocess.CalledProcessError as e:
        logging.error(f'Error listing {directory}: {e}')
        return []


def find_catalogs(client, entries, vos_path, local_path, file_pattern):
    for entry in tqdm(entries):
        logging.debug(f'Processing entry: {entry}')
        sub_dir_path = os.path.join(vos_path, entry)
        if client.isdir(sub_dir_path):
            local_file_path = os.path.join(local_path, entry, entry + file_pattern)
            logging.debug(f'Local file path: {local_file_path}')
            if os.path.exists(local_file_path):
                logging.info(f'Catalog {entry} already downloaded. Skipping.')
                continue
            files = list_vospace(sub_dir_path)
            logging.debug(f'Found files: {files}')
            for file in files:
                logging.debug(f'Processing file: {file}')
                file_path = os.path.join(sub_dir_path, file)
                if file.endswith(file_pattern):
                    # create the local directory if it doesn't exist to maintian the same structure
                    os.makedirs(os.path.join(local_path, entry), exist_ok=True)
                    local_file_path = os.path.join(local_path, entry, file)
                    if os.path.exists(local_file_path):
                        logging.info(f'File {file} already downloaded. Skipping.')
                        continue
                    if client.isfile(file_path):
                        logging.info(f'Found catalog: {file}, starting download..')
                        download_file(file_path, local_file_path)


def download_file(vospace_path, local_path):
    """Download a specific file using vcp"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        subprocess.run(['vcp', vospace_path, local_path], check=True)
        logging.info(f'Downloaded: {local_path}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Failed to download {vospace_path}: {e}')


def main():
    # VOSpace directory to start listing from
    base_vospace_dir = 'arc:projects/unions/catalogues/unions/GAaP_photometry/UNIONS2000'
    local_dir = '/home/heesters/projects/def-sfabbro/heesters/data/unions/catalogs/GAaP/UNIONS2000'

    client = vos.Client()

    # List the contents of the base directory recursively
    entries = list_vospace(base_vospace_dir)
    file_pattern = '_ugriz_photoz_ext.cat'

    if entries is not None:
        logging.info(f'Found {len(entries)} entries in the directory.')
        print(entries[:30])

    if entries:
        # Process the directory entries
        find_catalogs(client, entries, base_vospace_dir, local_dir, file_pattern)


if __name__ == '__main__':
    main_directory = '/home/heesters/projects/def-sfabbro/heesters/github/TileSlicer'
    log_directory = os.path.join(main_directory, 'logs/')
    os.makedirs(log_directory, exist_ok=True)
    setup_logging(log_directory, __file__, logging_level=logging.INFO)
    logging.info('Downloading GAaP catalogs from VOSpace...')

    main()

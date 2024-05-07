import glob
import logging
import os
from itertools import groupby

from tqdm import tqdm

from data_utils import create_master_cat_from_file, extract_numbers, update_h5_labels_parallel

# define the root directory
main_directory = '/arc/home/heestersnick/tileslicer/'
# define the data directory
data_directory = '/arc/projects/unions/ssl/data/'
# define the table directory
table_directory = os.path.join(main_directory, 'tables/')
# define UNIONS table directory
unions_table_directory = '/arc/projects/unions/catalogues/'
# define the path to the UNIONS detection catalogs
unions_detection_directory = os.path.join(
    unions_table_directory, 'unions/GAaP_photometry/UNIONS2000/'
)
# define the path to the catalog containing redshifts and classes
redshift_class_catalog = os.path.join(
    unions_table_directory, 'redshifts/redshifts-2024-05-07.parquet'
)
# define where the cutouts are saved
cutout_directory = os.path.join(data_directory, 'processed/unions-cutouts/cutouts2024/')
# define the path to the catalog containing known lenses
lens_catalog = os.path.join(table_directory, 'known_lenses.parquet')
# define the path to the master catalog
master_catalog = os.path.join(table_directory, 'cutout_catalog_master.parquet')

# define where the logs should be saved
log_dir = os.path.join(main_directory, 'logs/')
os.makedirs(log_dir, exist_ok=True)

# define the logger
log_file_name = 'label_update.log'
log_file_path = os.path.join(log_dir, log_file_name)

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler(),  # Add this line to also log to the console
    ],
)

# create/update a master catalog containing all cut out objects
update_master_catalog = False


def main(cutout_dir, table_dir, cat_master, unions_det_dir, z_class_cat, lens_cat, update_master):
    # get list of file paths
    file_paths = sorted(set(glob.glob(os.path.join(cutout_dir, '*.h5'))))
    # get list of file names
    file_names = [os.path.basename(file_path) for file_path in file_paths]
    # sort file names by tile numbers
    file_names.sort(key=extract_numbers)
    # print(file_names.index('351_272_224x224_ugri_batch_5.h5'))
    grouped_files = {key: list(group) for key, group in groupby(file_names, key=extract_numbers)}

    # Process each group of file paths
    for key, files in tqdm(grouped_files.items()):
        logging.info(f'Updating labels for tile {key}:')
        file_path_list = [os.path.join(cutout_dir, file) for file in files]
        update_h5_labels_parallel(key, file_path_list, unions_det_dir, z_class_cat, lens_cat)
        if update_master:
            logging.info(f'Adding detections in tile {key} to the master catalog.')
            create_master_cat_from_file(cat_master, table_dir, file_path_list, key)
            logging.info(f'Added detections in tile {key} to the master catalog.')


if __name__ == '__main__':
    main(
        cutout_directory,
        table_directory,
        master_catalog,
        unions_detection_directory,
        redshift_class_catalog,
        lens_catalog,
        update_master_catalog,
    )

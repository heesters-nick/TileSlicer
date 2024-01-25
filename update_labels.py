import glob
import logging
import os

from tqdm import tqdm

from utils import update_h5_labels

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
    unions_table_directory, 'redshifts/redshifts-2024-01-04.parquet'
)
# define where the cutouts are saved
cutout_directory = os.path.join(data_directory, 'processed/unions-cutouts/cutouts2024/')
# define the path to the catalog containing known lenses
lens_catalog = os.path.join(table_directory, 'known_lenses.parquet')

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


def main(cutout_dir, unions_det_dir, z_class_cat, lens_cat):
    # get list of file paths
    file_paths = sorted(set(glob.glob(os.path.join(cutout_dir, '*.h5'))))
    # file_names = [os.path.basename(file_path) for file_path in file_paths]
    # print(file_names.index('351_272_224x224_ugri_batch_4.h5'))
    # loop over file paths
    for file_path in tqdm(file_paths):
        # if '247_255' in file_path:
        update_h5_labels(file_path, unions_det_dir, z_class_cat, lens_cat)


if __name__ == '__main__':
    main(cutout_directory, unions_detection_directory, redshift_class_catalog, lens_catalog)

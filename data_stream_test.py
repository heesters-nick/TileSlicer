import logging
import os
import random
import time

import numpy as np
import pandas as pd
from vos import Client

from data_stream import DataStream

client = Client()

band_dict = {
    'cfis-u': {
        'name': 'CFIS',
        'band': 'u',
        'vos': 'vos:cfis/tiles_DR5/',
        'suffix': '.u.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
    'whigs-g': {
        'name': 'calexp-CFIS',
        'band': 'g',
        'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme/',
        'suffix': '.fits',
        'delimiter': '_',
        'fits_ext': 1,
        'zfill': 0,
    },
    'cfis_lsb-r': {
        'name': 'CFIS_LSB',
        'band': 'r',
        'vos': 'vos:cfis/tiles_LSB_DR5/',
        'suffix': '.r.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
    'ps-i': {
        'name': 'PS-DR3',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR3/tiles/',
        'suffix': '.i.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
    'wishes-z': {
        'name': 'WISHES',
        'band': 'z',
        'vos': 'vos:cfis/wishes_1/coadd/',
        'suffix': '.z.fits',
        'delimiter': '.',
        'fits_ext': 1,
        'zfill': 0,
    },
}


# retrieve from the VOSpace and update the currently available tiles; takes some time to run
update_tiles = False
# build kd tree with updated tiles otherwise use the already saved tree
if update_tiles:
    build_new_kdtree = True
else:
    build_new_kdtree = False
# return the number of available tiles that are available in at least 5, 4, 3, 2, 1 bands
at_least = False
# show stats on currently available tiles, remember to update
show_tile_statistics = False
# show number of tiles available including this band
combinations_with_band = 'cfis_lsb-r'
# print per tile availability
print_per_tile_availability = False
# use UNIONS catalogs to make the cutouts
with_unions_catalogs = False
# download the tiles
download_tiles = True
# Plot cutouts from one of the tiles after execution
with_plot = True
# Plot a random cutout from one of the tiles after execution else plot all cutouts
plot_random_cutout = True
# Show plot
show_plot = False
# Save plot
save_plot = True

# paths
# define the root directory
main_directory = '/arc/home/heestersnick/tileslicer/'
data_directory = '/arc/projects/unions/ssl/data/'
table_directory = os.path.join(main_directory, 'tables/')
os.makedirs(table_directory, exist_ok=True)
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
# define the path to the catalog containing known lenses
lens_catalog = os.path.join(table_directory, 'known_lenses.parquet')
# define the path to the master catalog that accumulates information about the cut out objects
catalog_master = os.path.join(table_directory, 'cutout_cat_master.parquet')
# define the path to the catalog containing known dwarf galaxies
dwarf_catalog = os.path.join(table_directory, 'all_known_dwarfs_processed.csv')
# define path to file containing the processed h5 files
processed_file = os.path.join(table_directory, 'processed.txt')
# define catalog file
catalog_file = 'all_known_dwarfs.csv'
catalog_script = pd.read_csv(os.path.join(table_directory, catalog_file))
# define the keys for ra, dec, and id in the catalog
ra_key_script, dec_key_script, id_key_script = 'ra', 'dec', 'ID'
# define where the information about the currently available tiles should be saved
tile_info_directory = os.path.join(main_directory, 'tile_info/')
os.makedirs(tile_info_directory, exist_ok=True)
# define where the tiles should be saved
download_directory = os.path.join(data_directory, 'raw/tiles/tiles2024/')
os.makedirs(download_directory, exist_ok=True)
# define where the cutouts should be saved
# cutout_directory = os.path.join(data_directory, 'processed/unions-cutouts/cutouts2024/')
# os.makedirs(cutout_directory, exist_ok=True)
cutout_directory = os.path.join(main_directory, 'cutouts/')
os.makedirs(cutout_directory, exist_ok=True)
# define where figures should be saved
figure_directory = os.path.join(main_directory, 'figures/')
os.makedirs(figure_directory, exist_ok=True)
# define where the logs should be saved
log_dir = os.path.join(main_directory, 'logs/')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('data_stream_test.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


band_constraint = 5  # define the minimum number of bands that should be available for a tile
cutout_size = 224
number_objects = 30000
num_workers = 5
queue_size = 3


def simulated_training_step(item):
    """Stand-in for your actual training process. Takes an item from the dataset"""
    logging.info(
        f'Cutout shape: {item[0].shape}, cutout datatype: {item[0].dtype}. Length catalog: {len(item[1])}.'
    )
    if np.any(item[0] != 0):
        logging.info(f'Cutout stack for tile {item[1]['tile'].values[0]} is not empty.')
    process_time = random.uniform(50, 80)  # Random training duration
    time.sleep(process_time)
    logging.info(f'Processed: {item[1]['tile'].values[0]} (simulated {process_time:.2f}s delay)')


def main(
    tile_info_dir,
    unions_det_dir,
    band_constr,
    download_dir,
    in_dict,
    cutout_size,
    at_least_key,
    dwarf_cat,
    z_class_cat,
    lens_cat,
    num_objects,
    show_stats,
    cutout_workers,
    queue_size,
):
    dataset = DataStream(
        tile_info_dir,
        unions_det_dir,
        band_constr,
        download_dir,
        in_dict,
        cutout_size,
        at_least_key,
        dwarf_cat,
        z_class_cat,
        lens_cat,
        num_objects,
        show_stats,
        cutout_workers,
        queue_size,
    )  # Initialize your dataset

    num_iterations = 10  # How many batches you want to simulate
    if dataset.preload():
        logging.info('Preload finished.')
        for _ in range(num_iterations):
            try:
                cutouts, catalog = dataset.__next__()
                simulated_training_step((cutouts, catalog))

            except StopIteration:
                print('Dataset Exhausted.')
                break


if __name__ == '__main__':
    main(
        tile_info_directory,
        unions_detection_directory,
        band_constraint,
        download_directory,
        band_dict,
        cutout_size,
        at_least,
        dwarf_catalog,
        redshift_class_catalog,
        lens_catalog,
        number_objects,
        show_tile_statistics,
        num_workers,
        queue_size,
    )

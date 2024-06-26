import logging
import os
import sys

import pandas as pd
from vos import Client

from data_stream import TileStream
from data_utils import setup_logging, update_df_tile_stats, update_processed

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
    'ps-z': {
        'name': 'DR4',
        'band': 'ps-z',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
        'suffix': '.z.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
}

# define the bands to consider
considered_bands = ['cfis-u', 'whigs-g', 'cfis_lsb-r', 'ps-i', 'wishes-z']
# create a dictionary with the bands to consider
band_dict_incl = {key: band_dict.get(key) for key in considered_bands}


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
show_tile_statistics = True
# show number of tiles available including this band
combinations_with_band = 'cfis_lsb-r'
# print per tile availability
print_per_tile_availability = False
# use UNIONS catalogs to make the cutouts
with_unions_catalogs = False
# download the tiles
download_tiles = True
# Plot cutouts from one of the tiles after execution
with_plot = False
# Plot a random cutout from one of the tiles after execution else plot all cutouts
plot_random_cutout = True
# Show plot
show_plot = False
# Save plot
save_plot = True

platform = 'cedar'  #'CANFAR'
if platform == 'CANFAR':
    root_dir_main = '/arc/home/ashley/SSL/git/'
    root_dir_data = '/arc/projects/unions/'
    root_dir_downloads = (
        '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/'
    )
else:  # assume compute canada for now
    root_dir_main = '/home/heesters/projects/def-sfabbro/heesters/github'
    root_dir_data_ashley = '/home/heesters/projects/def-sfabbro/a4ferrei/data'
    root_dir_data = '/home/heesters/projects/def-sfabbro/heesters/data'

# paths
# define the root directory
main_directory = os.path.join(root_dir_main, 'TileSlicer')
data_directory = root_dir_data
table_directory = os.path.join(main_directory, 'tables')
os.makedirs(table_directory, exist_ok=True)
# define UNIONS table directory
unions_table_directory = os.path.join(root_dir_data_ashley, 'catalogues')
# define the path to the UNIONS detection catalogs
unions_detection_directory = os.path.join(
    unions_table_directory, 'unions/GAaP_photometry/UNIONS2000/'
)
# define the path to the catalog containing redshifts and classes
redshift_class_catalog = os.path.join(
    unions_table_directory, 'redshifts/redshifts-2024-05-07.parquet'
)
# define the path to the catalog containing known lenses
lens_catalog = os.path.join(table_directory, 'known_lenses.parquet')
# define the path to the master catalog that accumulates information about the cut out objects
catalog_master = os.path.join(table_directory, 'cutout_cat_master.parquet')
# define the path to the catalog containing known dwarf galaxies
dwarf_catalog = os.path.join(table_directory, 'all_known_dwarfs_processed.csv')
# define path to file containing the processed h5 files
processed_file = os.path.join(table_directory, 'processed_stats.txt')
# define path to dataframe used to accumulate tile stats
df_tile_statistics = os.path.join(table_directory, 'tile_statistics.csv')
# create empty file if it does not already exist
if not os.path.exists(processed_file):
    with open(processed_file, 'w') as file:
        file.write('')
# define catalog file
catalog_file = 'all_known_dwarfs.csv'
catalog_script = pd.read_csv(os.path.join(table_directory, catalog_file))
# define the keys for ra, dec, and id in the catalog
ra_key_script, dec_key_script, id_key_script = 'ra', 'dec', 'ID'
# define where the information about the currently available tiles should be saved
tile_info_directory = os.path.join(main_directory, 'tile_info/')
os.makedirs(tile_info_directory, exist_ok=True)
# define where the tiles should be saved
download_directory = os.path.join(data_directory, 'unions/tiles')
os.makedirs(download_directory, exist_ok=True)
# define where the cutouts should be saved
cutout_directory = os.path.join(data_directory, 'cutouts/')
os.makedirs(cutout_directory, exist_ok=True)
# define where figures should be saved
figure_directory = os.path.join(main_directory, 'figures/')
os.makedirs(figure_directory, exist_ok=True)
# define where the logs should be saved
log_directory = os.path.join(main_directory, 'logs/')
os.makedirs(log_directory, exist_ok=True)

band_constraint = 5  # define the minimum number of bands that should be available for a tile
cutout_size = 224
number_objects = 10000  # give number of objects per tile that should be processed or say 'all'
num_stats_workers = 5  # number of threads for cutout creation
num_download_workers = 5  # number of threads for tile download
queue_size = 10  # max queue size
logging_level = logging.DEBUG
exclude_processed_tiles = True  # exclude already processed tiles from training


def main(
    update,
    tile_info_dir,
    unions_det_dir,
    band_constr,
    download_dir,
    in_dict,
    at_least_key,
    show_stats,
    stats_workers,
    download_workers,
    queue_size,
    processed,
    df_tile_stats,
    exclude_processed,
    log_dir,
    log_level,
):
    setup_logging(log_dir, __file__, logging_level=log_level)

    dataset = TileStream(
        update,
        tile_info_dir,
        unions_det_dir,
        band_constr,
        download_dir,
        in_dict,
        at_least_key,
        show_stats,
        stats_workers,
        download_workers,
        queue_size,
        processed,
        exclude_processed,
    )  # Initialize dataset
    num_iterations = 1000000  # How many training steps should be simulated
    # Prefill the queue to create a buffer
    if dataset.preload():
        for _ in range(num_iterations):
            try:
                tile_stats, tile = dataset.__next__()  # pull out the next item
                logging.debug(f'Got stats for tile {tile}.')
                update_df_tile_stats(df_tile_stats, tile_stats)
                update_processed(str(tile), processed)
                del tile_stats  # cleanup
                del tile

            except KeyboardInterrupt:
                sys.exit('Stopping the script due to KeyboardInterrupt.')

    logging.info('Max number of iterations reached. Stopping script.')


if __name__ == '__main__':
    main(
        update_tiles,
        tile_info_directory,
        unions_detection_directory,
        band_constraint,
        download_directory,
        band_dict_incl,
        at_least,
        show_tile_statistics,
        num_stats_workers,
        num_download_workers,
        queue_size,
        processed_file,
        df_tile_statistics,
        exclude_processed_tiles,
        log_directory,
        logging_level,
    )

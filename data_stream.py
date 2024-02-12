import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import Pool

import numba as nb
import numpy as np
import pandas as pd
from astropy.io import fits
from vos import Client

from kd_tree import build_tree
from plotting import plot_cutout
from tile_cutter import (
    download_tile_for_bands_parallel,
    tiles_from_unions_catalogs,
)
from utils import (
    TileAvailability,
    add_labels,
    extract_tile_numbers,
    load_available_tiles,
    read_dwarf_cat,
    read_h5,
    read_unions_cat,
    update_available_tiles,
)

client = Client()

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .

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

# define the logger
log_file_name = 'datastream.log'
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

### tile parameters ###
band_constraint = 5  # define the minimum number of bands that should be available for a tile
tile_batch_size = 7  # number of tiles to process in parallel
object_batch_size = 5000  # number of objects to process at a time
cutout_size = np.float32(224)
cutout_size = 224
num_workers = 14  # specifiy the number of parallel workers following machine capabilities


def open_fits_files_concurrently(tile_dir, fits_filenames, fits_ext):
    fits_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_filename = {
            executor.submit(
                fits.open, os.path.join(tile_dir, fits_filename), memmap=True
            ): fits_filename
            for fits_filename in fits_filenames
        }
        for future in as_completed(future_to_filename):
            fits_filename = future_to_filename[future]
            band_ext = fits_ext[fits_filenames.index(fits_filename)]
            try:
                hdul = future.result()
                if (
                    hdul is not None and len(hdul) > 0
                ):  # Check if hdul is not None and contains at least one HDU
                    fits_data[fits_filename] = hdul[band_ext].data.astype(np.float32)
                else:
                    logging.warning(
                        f'Empty or invalid HDU for FITS file {fits_filename}. Skipping.'
                    )
            except FileNotFoundError:
                logging.error(f'File {fits_filename} not found.')
    return fits_data


def open_fits_files_sequentially(tile_dir, fits_filenames, fits_ext):
    fits_data = {}
    for fits_filename in fits_filenames:
        try:
            with fits.open(
                os.path.join(tile_dir, fits_filename), memmap=True, mode='readonly'
            ) as hdul:
                if (
                    hdul is not None and len(hdul) > 0
                ):  # Check if hdul is not None and contains at least one HDU
                    band_ext = fits_ext[fits_filenames.index(fits_filename)]
                    fits_data[fits_filename] = hdul[band_ext].data.astype(np.float32)
                else:
                    logging.warning(
                        f'Empty or invalid HDU for FITS file {fits_filename}. Skipping.'
                    )
        except FileNotFoundError:
            logging.error(f'File {fits_filename} not found.')
    return fits_data


@nb.njit(nb.float32[:, :](nb.float32[:, :], nb.int32, nb.int32, nb.int32, nb.float32[:, :]))
def cutout2d(data_, x, y, size, cutout_in):
    y_large, x_large = data_.shape
    y_large, x_large = nb.int32(y_large), nb.int32(x_large)
    height, width = size, size

    y_start = max(0, y - height // 2)
    y_end = min(y_large, y + (height + 1) // 2)

    x_start = max(0, x - width // 2)
    x_end = min(x_large, x + (width + 1) // 2)

    if y_start >= y_end or x_start >= x_end:
        raise ValueError('No overlap between the small and large array.')

    # cutout = np.zeros((size, size), dtype=data.dtype)
    cutout_in[
        y_start - y + height // 2 : y_end - y + height // 2,
        x_start - x + width // 2 : x_end - x + width // 2,
    ] = data_[y_start:y_end, x_start:x_end]

    return cutout_in


def cutout_one_band(tile, obj_in_tile, download_dir, in_dict, size, band):
    cutouts_for_band = np.zeros((len(obj_in_tile), size, size), dtype=np.float32)
    tile_dir = download_dir + f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
    prefix = in_dict[band]['name']
    suffix = in_dict[band]['suffix']
    delimiter = in_dict[band]['delimiter']
    fits_ext = in_dict[band]['fits_ext']
    zfill = in_dict[band]['zfill']
    tile_fitsfilename = f'{prefix}{delimiter}{str(tile[0]).zfill(zfill)}{delimiter}{str(tile[1]).zfill(zfill)}{suffix}'
    size = np.int32(size)
    try:
        fits_start = time.time()
        with fits.open(os.path.join(tile_dir, tile_fitsfilename), memmap=True) as hdul:
            data = hdul[fits_ext].data.astype(np.float32)  # type: ignore
            logging.info(f'Opened {tile_fitsfilename}. Took {np.round(time.time()-fits_start, 2)}')
            cutout_empty = np.zeros((size, size), dtype=np.float32)
            xs, ys = (
                np.floor(obj_in_tile.x.values + 0.5).astype(np.int32),
                np.floor(obj_in_tile.y.values + 0.5).astype(np.int32),
            )
            cutout_start = time.time()
            for i, (x, y) in enumerate(zip(xs, ys)):
                # cutouts_for_band[i] = make_cutout(data, x, y, size)
                cutouts_for_band[i] = cutout2d(data, x, y, size, cutout_empty)
            logging.info(
                f'Finished cutting {len(xs)} objects for {band} in {np.round(time.time()-cutout_start, 2)} seconds.'
            )
    except FileNotFoundError:
        logging.info(f'File {tile_fitsfilename} not found.')
        return None

    return cutouts_for_band


def cutout_bands_parallel_cf(tile, in_dict, download_dir, obj_in_tile, size):
    n_bands = len(in_dict)
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)
    parallel_start = time.time()
    with ProcessPoolExecutor() as executor:
        # Dictionary mapping each future to the corresponding band
        future_to_band = {
            executor.submit(
                cutout_one_band, tile, obj_in_tile, download_dir, in_dict, size, band
            ): band
            for band in in_dict.keys()
        }
        for future in as_completed(future_to_band):
            band = future_to_band[future]
            band_idx = list(in_dict.keys()).index(band)
            try:
                result = future.result()
                if result is not None:
                    final_cutouts[:, band_idx] = result
            except Exception as e:
                logging.exception(f'Failed to process band {band} for tile {tile}: {str(e)}')

    parallel_end = time.time()
    logging.info(
        f'Finished cutting for all bands in {np.round(parallel_end-parallel_start, 2)} seconds.'
    )
    return final_cutouts


def cutout_bands_parallel_mp(tile, in_dict, download_dir, obj_in_tile, size):
    n_bands = len(in_dict)
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)
    parallel_start = time.time()
    # Create a Pool with the number of available CPUs
    with Pool() as pool:
        # Map the process_band function to each band
        results = pool.starmap(
            cutout_one_band,
            [(tile, obj_in_tile, download_dir, in_dict, size, band) for band in in_dict.keys()],
        )

        # Fill final_cutouts with the results
        for band_idx, result in enumerate(results):
            if result is not None:
                final_cutouts[:, band_idx] = result

    parallel_end = time.time()
    logging.info(
        f'Finished cutting for all bands in {np.round(parallel_end-parallel_start, 2)} seconds.'
    )

    return final_cutouts


def cutout_bands_sequential(tile, in_dict, download_dir, obj_in_tile, size):
    n_bands = len(in_dict)
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)

    # Iterate over each band sequentially
    for band_idx, band in enumerate(in_dict.keys()):
        try:
            result = cutout_one_band(tile, obj_in_tile, download_dir, in_dict, size, band)
            if result is not None:
                final_cutouts[:, band_idx] = result
        except Exception as e:
            logging.exception(f'Failed to process band {band} for tile {tile}: {str(e)}')

    return final_cutouts


def cutout_bands_sequential_new(tile, in_dict, download_dir, obj_in_tile, size):
    n_bands = len(in_dict)
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)

    # Open FITS files concurrently
    tile_dir = os.path.join(download_dir, f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}')
    fits_filenames = [
        f'{in_dict[band]["name"]}{in_dict[band]["delimiter"]}{str(tile[0]).zfill(in_dict[band]["zfill"])}{in_dict[band]["delimiter"]}{str(tile[1]).zfill(in_dict[band]["zfill"])}{in_dict[band]["suffix"]}'
        for band in in_dict.keys()
    ]
    fits_extensions = [in_dict[band]['fits_ext'] for band in in_dict.keys()]
    open_start = time.time()
    # fits_data = open_fits_files_concurrently(tile_dir, fits_filenames, fits_extensions)
    fits_data = open_fits_files_sequentially(tile_dir, fits_filenames, fits_extensions)
    logging.info(
        f'Finished opening all fits files in {np.round(time.time()-open_start, 2)} seconds.'
    )
    for band_idx, (band, data) in enumerate(fits_data.items()):
        final_cutouts[:, band_idx] = cutout_one_band_new(data, obj_in_tile, size, band)

    return final_cutouts


def cutout_one_band_new(data, obj_in_tile, size, band):
    cutouts_for_band = np.zeros((len(obj_in_tile), size, size), dtype=np.float32)
    cutout_empty = np.zeros((size, size), dtype=np.float32)
    xs, ys = (
        np.floor(obj_in_tile.x.values + 0.5).astype(np.int32),
        np.floor(obj_in_tile.y.values + 0.5).astype(np.int32),
    )
    cutout_start = time.time()
    for i, (x, y) in enumerate(zip(xs, ys)):
        cutouts_for_band[i] = cutout2d(data, x, y, size, cutout_empty)

    logging.info(
        f'Finished cutting {len(xs)} objects for {band} in {np.round(time.time()-cutout_start, 2)} seconds.'
    )
    return cutouts_for_band


# Define cutout_bands_parallel function with concurrent file opening
def cutout_bands_parallel_new(tile, in_dict, download_dir, obj_in_tile, size):
    n_bands = len(in_dict)
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)

    # Open FITS files concurrently
    tile_dir = os.path.join(download_dir, f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}')
    fits_filenames = [
        f'{in_dict[band]["name"]}{in_dict[band]["delimiter"]}{str(tile[0]).zfill(in_dict[band]["zfill"])}{in_dict[band]["delimiter"]}{str(tile[1]).zfill(in_dict[band]["zfill"])}{in_dict[band]["suffix"]}'
        for band in in_dict.keys()
    ]
    fits_extensions = [in_dict[band]['fits_ext'] for band in in_dict.keys()]
    open_start = time.time()
    # fits_data = open_fits_files_concurrently(tile_dir, fits_filenames, fits_extensions)
    fits_data = open_fits_files_sequentially(tile_dir, fits_filenames, fits_extensions)
    logging.info(
        f'Finished opening all fits files in {np.round(time.time()-open_start, 2)} seconds.'
    )

    # Perform parallel cutout creation
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Dictionary mapping each future to the corresponding band
        future_to_band = {
            executor.submit(cutout_one_band_new, data, obj_in_tile, size, band): i
            for i, (band, data) in enumerate(fits_data.items())
        }
        for future in as_completed(future_to_band):
            band_idx = future_to_band[future]
            try:
                result = future.result()
                if result is not None:
                    final_cutouts[:, band_idx] = result
            except Exception as e:
                logging.exception(f'Failed to process band {band_idx}: {str(e)}')

    return final_cutouts


def main(
    dwarf_cat,
    z_class_cat,
    lens_cat,
    cat_master,
    processed,
    tile_info_dir,
    in_dict,
    comb_w_band,
    at_least_key,
    band_constr,
    download_dir,
    cutout_dir,
    figure_dir,
    table_dir,
    unions_det_dir,
    size,
    workers,
    update,
    show_stats,
    dl_tiles,
    build_kdtree,
    w_plot,
    show_plt,
    save_plt,
):
    scrip_start = time.time()

    if update:
        update_available_tiles(tile_info_dir)

    # extract the tile numbers from the available tiles
    u, g, lsb_r, i, z = extract_tile_numbers(load_available_tiles(tile_info_dir))
    all_bands = [u, g, lsb_r, i, z]
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict, at_least_key)
    # build the kd tree
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats(band=comb_w_band)
    # get the tiles to cut out from the unions catalogs
    _, tiles_x_bands = tiles_from_unions_catalogs(availability, unions_det_dir, band_constr)
    if dl_tiles:
        logging.info('Downloading the tiles in the available bands..')
        for tile in tiles_x_bands:
            start_download = time.time()
            if download_tile_for_bands_parallel(availability, tile, in_dict, download_dir):
                logging.info(
                    f'Tile downloaded in all available bands. Took {np.round(time.time() - start_download, 2)} seconds.'
                )
            else:
                logging.info(f'Tile {tile} failed to download.')

            avail_bands = ''.join(availability.get_availability(tile)[0])
            save_path = os.path.join(
                cutout_dir,
                f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}_{size}x{size}_{avail_bands}.h5',
            )

            # get objects to cut out
            obj_in_tile = read_unions_cat(unions_det_dir, tile)
            # add tile numbers to object dataframe
            obj_in_tile['tile'] = str(tile)
            # add available bands to object dataframe
            obj_in_tile['bands'] = str(avail_bands)
            # load the dwarf galaxies in the tile
            dwarfs_in_tile = read_dwarf_cat(dwarf_cat, tile)
            # add labels to the objects in the tile
            obj_in_tile = add_labels(obj_in_tile, dwarfs_in_tile, z_class_cat, lens_cat)

            # only cutout part of the objects for testing
            obj_in_tile = obj_in_tile[:20000].reset_index(drop=True)

            cutting_start = time.time()
            cutout = cutout_bands_sequential_new(tile, in_dict, download_dir, obj_in_tile, size)
            logging.info(
                f'Cutting finished. Took {np.round(time.time()-cutting_start, 2)} seconds.'
            )
            logging.info(f'Start to cutouts done took {np.round(time.time()-scrip_start, 2)}')

            # save_to_h5(
            #     cutout,
            #     tile,
            #     obj_in_tile['ID'].values,
            #     obj_in_tile['ra'].values,
            #     obj_in_tile['dec'].values,
            #     obj_in_tile['mag_r'].values,
            #     obj_in_tile['class'].values,
            #     obj_in_tile['zspec'].values,
            #     obj_in_tile['lsb'].values,
            #     obj_in_tile['lens'].values,
            #     save_path,
            # )

            cutout = None

            tile_folder = os.path.join(
                download_dir, f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
            )
            if os.path.exists(tile_folder):
                logging.info(f'Cutting done, deleting raw data from tile {tile}.')
                shutil.rmtree(tile_folder)

            # plot all cutouts or just a random one
            if with_plot:
                if plot_random_cutout:
                    logging.info(f'Plotting cutout of random object in tile: {tile}.')
                    cutout_from_file = read_h5(save_path)
                    random_obj_index = np.random.randint(0, cutout_from_file['images'].shape[0])
                    # plot a random object from the stack of cutouts
                    plot_cutout(
                        cutout_from_file,
                        in_dict,
                        figure_dir,
                        random_obj_index,
                        show_plot=show_plt,
                        save_plot=save_plt,
                    )
                else:
                    plot_cutout(cutout, in_dict, figure_dir, show_plot=show_plt, save_plot=save_plt)
            break


if __name__ == '__main__':
    # define the arguments for the main function
    arg_dict_main = {
        'dwarf_cat': dwarf_catalog,
        'z_class_cat': redshift_class_catalog,
        'lens_cat': lens_catalog,
        'cat_master': catalog_master,
        'processed': processed_file,
        'tile_info_dir': tile_info_directory,
        'in_dict': band_dict,
        'comb_w_band': combinations_with_band,
        'at_least_key': at_least,
        'band_constr': band_constraint,
        'download_dir': download_directory,
        'cutout_dir': cutout_directory,
        'figure_dir': figure_directory,
        'table_dir': table_directory,
        'unions_det_dir': unions_detection_directory,
        'size': cutout_size,
        'workers': num_workers,
        'update': update_tiles,
        'show_stats': show_tile_statistics,
        'dl_tiles': download_tiles,
        'build_kdtree': build_new_kdtree,
        'w_plot': with_plot,
        'show_plt': show_plot,
        'save_plt': save_plot,
    }

    start = time.time()
    main(**arg_dict_main)
    end = time.time()
    elapsed = end - start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = (
        elapsed_string.split(':')[0],
        elapsed_string.split(':')[1],
        elapsed_string.split(':')[2],
    )
    logging.info(
        f'Done! Execution took {hours} hours, {minutes} minutes, and {np.round(float(seconds),2)} seconds.'
    )

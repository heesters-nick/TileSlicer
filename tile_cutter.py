import argparse
import concurrent.futures
import glob
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs.utils import skycoord_to_pixel
from vos import Client

from kd_tree import TileWCS, build_tree, query_tree, relate_coord_tile
from plotting import plot_cutout
from utils import (
    TileAvailability,
    add_labels,
    extract_tile_numbers,
    get_numbers_from_folders,
    load_available_tiles,
    object_batch_generator,
    read_dwarf_cat,
    read_h5,
    read_processed,
    read_unions_cat,
    save_tile_cat,
    update_available_tiles,
    update_master_cat,
    update_processed,
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
plot_random_cutout = False
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
log_file_name = 'tilecutter.log'
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
band_constraint = 3  # define the minimum number of bands that should be available for a tile
tile_batch_size = 7  # number of tiles to process in parallel
object_batch_size = 5000  # number of objects to process at a time
cutout_size = 224
num_workers = 55  # specifiy the number of parallel workers following machine capabilities


def tile_finder(availability, catalog, coord_c, tile_info_dir, band_constr=5):
    """
    Finds tiles a list of objects are in.
    :param availability: object to retrieve available tiles
    :param catalog: object catalog
    :param coord_c: astropy SkyCoord object of the coordinates
    :param tile_info_dir: tile information directory
    :param band_constr: minimum number of bands that should be available
    :return: unique tiles the objects are in, tiles that meet the band constraint
    """
    available_tiles = availability.unique_tiles
    tiles_matching_catalog = np.empty(len(catalog), dtype=tuple)
    pix_coords = np.empty((len(catalog), 2), dtype=np.float64)
    bands = np.empty(len(catalog), dtype=object)
    n_bands = np.empty(len(catalog), dtype=np.int32)
    for i, obj_coord in enumerate(coord_c):
        tile_numbers = query_tree(
            available_tiles,
            np.array([obj_coord.ra.deg, obj_coord.dec.deg]),
            tile_info_dir,
        )
        tiles_matching_catalog[i] = tile_numbers
        # check how many bands are available for this tile
        bands_tile, band_idx_tile = availability.get_availability(tile_numbers)
        bands[i], n_bands[i] = ''.join(bands_tile), len(band_idx_tile)
        if (not bands_tile) or (tile_numbers is None):
            pix_coords[i] = np.nan, np.nan
            continue
        wcs = TileWCS()
        wcs.set_coords(relate_coord_tile(nums=tile_numbers))
        pix_coord = skycoord_to_pixel(obj_coord, wcs.wcs_tile, origin=1)
        pix_coords[i] = pix_coord

    # add tile numbers and pixel coordinates to catalog
    catalog['tile'] = tiles_matching_catalog
    catalog['x'] = pix_coords[:, 0]
    catalog['y'] = pix_coords[:, 1]
    catalog['bands'] = bands
    catalog['n_bands'] = n_bands
    unique_tiles = list(set(tiles_matching_catalog))
    tiles_x_bands = [
        tile for tile in unique_tiles if len(availability.get_availability(tile)[1]) >= band_constr
    ]

    return unique_tiles, tiles_x_bands, catalog


def tiles_from_unions_catalogs(avail, unions_table_dir, band_constr):
    """
    Get list of tiles from UNIONS catalogs that meet the band constraint.

    Args:
        avail (TileAvailability): instance of the TileAvailability class
        unions_table_dir (str): directory where the UNIONS catalogs are located
        band_constr (int): band constraint, tile must be available in at least this many filters

    Returns:
        unique_tiles (list): list of unique tiles for which a catalog is available
        tiles_x_bands (list): list of unique tiles that meet the band constraint
    """
    tile_list = get_numbers_from_folders(unions_table_dir)

    unique_tiles = list(set(tile_list))

    tiles_x_bands = [
        tile for tile in unique_tiles if len(avail.get_availability(tile)[1]) >= band_constr
    ]

    return unique_tiles, tiles_x_bands


def download_tile_for_bands(availability, tile_numbers, in_dict, download_dir, method='api'):
    """
    Download a tile for the available bands.
    :param availability: object to retrieve available tiles
    :param tile_numbers: 2 three digit tile numbers
    :param in_dict: band dictionary containing the necessary info on the file properties
    :param download_dir: download directory
    :param method: choose between 'command' and 'api' for command line and client interaction with the VOSpace
    :return: True/False if the download was successful/failed
    """
    avail_idx = availability.get_availability(tile_numbers)[1]
    tile_dir = download_dir + f'{str(tile_numbers[0]).zfill(3)}_{str(tile_numbers[1]).zfill(3)}'
    for band in np.array(list(band_dict.keys()))[avail_idx]:
        vos_dir = in_dict[band]['vos']
        prefix = in_dict[band]['name']
        suffix = in_dict[band]['suffix']
        delimiter = in_dict[band]['delimiter']
        zfill = in_dict[band]['zfill']
        os.makedirs(tile_dir, exist_ok=True)
        tile_fitsfilename = f'{prefix}{delimiter}{str(tile_numbers[0]).zfill(zfill)}{delimiter}{str(tile_numbers[1]).zfill(zfill)}{suffix}'
        # use a temporary name while the file is downloading
        temp_name = '.'.join(tile_fitsfilename.split('.')[:-1]) + '_temp.fits'

        # Check if the directory exists, and create it if not
        if os.path.exists(os.path.join(tile_dir, tile_fitsfilename)):
            logging.info(f'File {tile_fitsfilename} was already downloaded.')
        else:
            logging.info(f'Downloading {tile_fitsfilename}..')
            try:
                if method == 'command':
                    # command line
                    os.system(
                        f'vcp -v {vos_dir + tile_fitsfilename} {os.path.join(tile_dir, temp_name)}'
                    )
                else:
                    # API
                    client.copy(
                        os.path.join(vos_dir, tile_fitsfilename),
                        os.path.join(tile_dir, temp_name),
                    )
                os.rename(
                    os.path.join(tile_dir, temp_name),
                    os.path.join(tile_dir, tile_fitsfilename),
                )
            except Exception as e:
                logging.exception(e)
                return False
    return True


def download_tile_one_band(tile_numbers, tile_fitsname, final_path, temp_path, vos_path, band):
    if os.path.exists(final_path):
        logging.info(f'File {tile_fitsname} was already downloaded for band {band}.')
        return True

    try:
        logging.info(f'Downloading {tile_fitsname} for band {band}...')
        start_time = time.time()
        result = subprocess.run(
            f'vcp -v {vos_path} {temp_path}', shell=True, stderr=subprocess.PIPE, text=True
        )

        result.check_returncode()

        os.rename(temp_path, final_path)
        logging.info(f'Successfully downloaded tile {tuple(tile_numbers)} for band {band}.')
        logging.info(f'Finished in {np.round((time.time()-start_time)/60, 3)} minutes.')
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f'Failed downloading tile {tuple(tile_numbers)} for band {band}.')
        logging.error(f'Subprocess error details: {e}')
        return False

    except FileNotFoundError:
        logging.error(f'Failed downloading tile {tuple(tile_numbers)} for band {band}.')
        logging.exception(f'Tile {tuple(tile_numbers)} not available in {band}.')
        return False

    except Exception as e:
        logging.error(f'Tile {tuple(tile_numbers)} in {band}: an unexpected error occurred: {e}')
        return False


def download_tile_for_bands_parallel(availability, tile_nums, in_dict, download_dir):
    avail_idx = availability.get_availability(tile_nums)[1]
    with ThreadPoolExecutor() as executor:
        # Create a list of futures for concurrent downloads
        futures = []
        for band in np.array(list(band_dict.keys()))[avail_idx]:
            vos_dir = in_dict[band]['vos']
            prefix = in_dict[band]['name']
            suffix = in_dict[band]['suffix']
            delimiter = in_dict[band]['delimiter']
            zfill = in_dict[band]['zfill']
            tile_dir = download_dir + f'{str(tile_nums[0]).zfill(3)}_{str(tile_nums[1]).zfill(3)}'
            os.makedirs(tile_dir, exist_ok=True)
            tile_fitsfilename = f'{prefix}{delimiter}{str(tile_nums[0]).zfill(zfill)}{delimiter}{str(tile_nums[1]).zfill(zfill)}{suffix}'
            temp_name = '.'.join(tile_fitsfilename.split('.')[:-1]) + '_temp.fits'
            temp_path = os.path.join(tile_dir, temp_name)
            final_path = os.path.join(tile_dir, tile_fitsfilename)
            vos_path = os.path.join(vos_dir, tile_fitsfilename)

            future = executor.submit(
                download_tile_one_band,
                tile_nums,
                tile_fitsfilename,
                final_path,
                temp_path,
                vos_path,
                band,
            )
            futures.append(future)

        # Wait for all downloads to complete
        for future in futures:
            future.result()

    return True


def make_cutout(data, x, y, size):
    """
    Creates an image cutout centered on the object.
    :param data: image data, 2d array
    :param x: x coordinate of the cutout center
    :param y: y coordinate of the cutout center
    :param size: cutout size in pixels
    :return: cutout, 2d array
    """
    # logging.info('Cutting..')
    img_cutout = Cutout2D(data, (x, y), size, mode='partial', fill_value=0).data

    if (
        np.count_nonzero(np.isnan(img_cutout)) >= 0.05 * size**2
        or np.count_nonzero(img_cutout) == 0
    ):
        return np.zeros((size, size))  # Don't use this cutout

    img_cutout[np.isnan(img_cutout)] = 0
    # logging.info('Cutting finished.')
    return img_cutout


def make_cutouts_all_bands(
    availability, tile, obj_in_tile, download_dir, in_dict, size, obj_batch_num=None
):
    """
    Loops over all five bands for a given tile, creates cutouts of the targets and adds them to the band dictionary.
    :param availability: object to retrieve available tiles
    :param tile: tile numbers
    :param obj_in_tile: dataframe containing the known objects in this tile
    :param download_dir: directory storing the tiles
    :param in_dict: band dictionary
    :param size: square cutout size in pixels
    :return: updated band dictionary containing cutout data
    """
    if obj_batch_num is not None:
        logging.info(f'Cutting out objects in tile {tile}, batch {obj_batch_num+1}.')
    else:
        logging.info(f'Cutting out objects in tile {tile}.')
    avail_idx = availability.get_availability(tile)[1]
    cutout = np.zeros((len(obj_in_tile), len(in_dict), size, size))
    tile_dir = download_dir + f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
    for j, band in enumerate(np.array(list(in_dict.keys()))[avail_idx]):
        prefix = in_dict[band]['name']
        suffix = in_dict[band]['suffix']
        delimiter = in_dict[band]['delimiter']
        fits_ext = in_dict[band]['fits_ext']
        zfill = in_dict[band]['zfill']
        tile_fitsfilename = f'{prefix}{delimiter}{str(tile[0]).zfill(zfill)}{delimiter}{str(tile[1]).zfill(zfill)}{suffix}'
        with fits.open(os.path.join(tile_dir, tile_fitsfilename), memmap=True) as hdul:
            data = hdul[fits_ext].data  # type: ignore
        for i, (x, y) in enumerate(zip(obj_in_tile.x.values, obj_in_tile.y.values)):
            cutout[i, j] = make_cutout(data, x, y, size)
            # if tile == (247, 255):
            #     logging.info(f'Cut {i}/{len(obj_in_tile.x.values)} objects.')

    if obj_batch_num is not None:
        logging.info(f'Finished cutting objects in tile {tile}, batch {obj_batch_num+1}.')

        if np.any(cutout != 0):
            logging.info(f'Cutout stack for tile {tile}, batch {obj_batch_num+1} is not empty.')
    else:
        logging.info(f'Finished cutting objects in tile {tile}.')
        if np.any(cutout != 0):
            logging.info(f'Cutout stack for tile {tile} is not empty.')

    return cutout


def save_to_h5(
    stacked_cutout,
    tile_numbers,
    ids,
    ras,
    decs,
    mag_r,
    class_label,
    z_label,
    lsb_label,
    lens_label,
    save_path,
):
    """
    Save cutout data including metadata to file.

    Args:
        stacked_cutout (numpy.ndarray): stacked numpy array of the image data in different bands
        tile_numbers (tuple): tile numbers
        ids (list): object IDs
        ras (numpy.ndarray): right ascension coordinate array
        decs (numpy.ndarray): declination coordinate array
        mag_r (numpy.ndarray): r-band magnitude array
        class_label (numpy.ndarray): class label array
        z_label (numpy.ndarray): redshift label array
        lsb_label (numpy.ndarray): LSB class label array
        lens_label (numpy.ndarray): lens class label array
        save_path (str): path to save the cutout

    Returns:
        None
    """
    logging.info(f'Saving file: {save_path}')
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(save_path, 'w', libver='latest') as hf:
        hf.create_dataset('images', data=stacked_cutout.astype(np.float32))
        hf.create_dataset('tile', data=np.asarray(tile_numbers), dtype=np.int32)
        hf.create_dataset('cfis_id', data=np.asarray(ids, dtype='S'), dtype=dt)
        hf.create_dataset('ra', data=ras.astype(np.float32))
        hf.create_dataset('dec', data=decs.astype(np.float32))
        hf.create_dataset('mag_r', data=mag_r.astype(np.float32))
        hf.create_dataset('class', data=class_label.astype(np.float32))
        hf.create_dataset('zspec', data=z_label.astype(np.float32))
        hf.create_dataset('lsb', data=lsb_label.astype(np.float32))
        hf.create_dataset('lens', data=lens_label.astype(np.float32))
    pass


def process_tile(
    availability,
    tile,
    catalog,
    dwarf_cat,
    z_class_cat,
    lens_cat,
    processed,
    id_key,
    ra_key,
    dec_key,
    cutout_dir,
    download_dir,
    table_dir,
    unions_table_dir,
    in_dict,
    size,
    w_unions_cats,
    obj_batch_size,
):
    """
    Process a tile, create cutouts in all bands, save cutouts and metadata to hdf5 file
    :param availability: object to retrieve available tiles
    :param tile: tile numbers
    :param catalog: object catalog
    :param dwarf_cat: dwarf catalog
    :param z_class_cat: redshift and class catalog
    :param lens_cat: lens catalog
    :param processed: list of processed tiles
    :param id_key: id key in the catalog
    :param ra_key: ra key in the catalog
    :param dec_key: dec key in the catalog
    :param cutout_dir: cutout directory
    :param download_dir: tile directory
    :param table_dir: table directory
    :param unions_table_dir: unions table directory
    :param in_dict: band dictionary
    :param size: cutout size
    :param w_unions_cats: use UNIONS catalogs
    :param obj_batch_size: number of objects to process at a time
    :return image cutout in available bands, array with shape: (n_bands, cutout_size, cutout_size)
    """
    avail_bands = ''.join(availability.get_availability(tile)[0])
    save_path = os.path.join(
        cutout_dir,
        f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}_{size}x{size}_{avail_bands}.h5',
    )
    base_path, extension = os.path.splitext(save_path)

    # initialize cutout
    cutout = None
    # initialize return variables
    n_cutouts, n_already_cutout, n_batches_processed, batch_nr = 0, 0, 0, 0
    all_already_processed = False

    if w_unions_cats:
        obj_in_tile = read_unions_cat(unions_table_dir, tile)
        dwarfs_in_tile = read_dwarf_cat(dwarf_cat, tile)
        obj_in_tile = add_labels(obj_in_tile, dwarfs_in_tile, z_class_cat, lens_cat)
        if obj_in_tile is None:
            logging.info(f'No objects cut out in tile {tile}.')
            return 0, 0, 0, 0
        obj_in_tile['tile'] = str(tile)
        obj_in_tile['bands'] = str(avail_bands)

        # count total number of cutouts created for this tile
        n_cutouts, n_already_cutout, n_batches_processed = 0, 0, 0
        all_already_processed = False
        # process in batches to avoid memory leakage
        for batch_nr, obj_batch in enumerate(object_batch_generator(obj_in_tile, obj_batch_size)):
            # check if the tile has already been processed
            tile_batch_name = f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}_{size}x{size}_{avail_bands}_batch_{batch_nr+1}'
            # if os.path.exists(f'{base_path}_batch_{batch_nr+1}{extension}'):
            if tile_batch_name in read_processed(processed):
                logging.info(f'Tile {tile} batch {batch_nr+1} has already been processed.')
                n_already_cutout += len(obj_batch)
                n_batches_processed += 1
                continue

            logging.info(
                f'Adding {np.count_nonzero(~np.isnan(obj_batch["zspec"]))} redshifts to the cutout.'
            )
            logging.info(
                f'Adding {np.count_nonzero(~np.isnan(obj_batch["class"]))} classes to the cutout.'
            )
            logging.info(
                f'Adding {np.count_nonzero(~np.isnan(obj_batch["lsb"]))} lsb objects to the cutout.'
            )
            logging.info(
                f'Adding {np.count_nonzero(~np.isnan(obj_batch["lens"]))} lens candidates to the cutout.'
            )

            cutout = make_cutouts_all_bands(
                availability, tile, obj_batch, download_dir, in_dict, size, batch_nr
            )

            save_to_h5(
                cutout,
                tile,
                obj_batch[id_key].values,
                obj_batch[ra_key].values,
                obj_batch[dec_key].values,
                obj_batch['mag_r'].values,
                obj_batch['class'].values,
                obj_batch['zspec'].values,
                obj_batch['lsb'].values,
                obj_batch['lens'].values,
                f'{base_path}_batch_{batch_nr+1}{extension}',
            )

            update_processed(tile_batch_name, processed)

            if not np.all(cutout == 0):
                n_cutouts += cutout.shape[0]

            # release memory
            cutout = None
    else:
        obj_in_tile = catalog.loc[catalog['tile'] == tile].reset_index(drop=True)
        dwarfs_in_tile = read_dwarf_cat(dwarf_cat, tile)
        if os.path.exists(save_path):
            logging.info(f'Tile {tile} has already been processed.')
            return 0, len(obj_in_tile), len(obj_in_tile), True

        obj_in_tile = add_labels(obj_in_tile, dwarfs_in_tile, z_class_cat, lens_cat)
        if obj_in_tile is None:
            logging.info(f'No objects cut out in tile {tile}.')
            return 0, 0, 0, 0

        cutout = make_cutouts_all_bands(
            availability, tile, obj_in_tile, download_dir, in_dict, size
        )
        # no r-band magnitude available for the dwarfs
        mag_r = np.nan * np.ones(len(obj_in_tile))
        save_to_h5(
            cutout,
            tile,
            obj_in_tile[id_key].values,
            obj_in_tile[ra_key].values,
            obj_in_tile[dec_key].values,
            mag_r,
            obj_in_tile['class'].values,
            obj_in_tile['zspec'].values,
            obj_in_tile['lsb'].values,
            obj_in_tile['lens'].values,
            save_path,
        )
        n_cutouts = cutout.shape[0]

    if w_unions_cats:
        if n_batches_processed == (batch_nr + 1):
            all_already_processed = True
            logging.info(
                f'All objects in tile {tile} have already been processed. Skipping catalog saving.'
            )
        else:
            # save catalog to temporary file
            save_tile_cat(table_dir, tile, obj_in_tile)

    return n_cutouts, n_already_cutout, len(obj_in_tile), all_already_processed


def process_tiles_in_batches(tile_list, batch_size):
    for i in range(0, len(tile_list), batch_size):
        yield tile_list[i : i + batch_size]


def main(
    cat_default,
    dwarf_cat,
    z_class_cat,
    lens_cat,
    cat_master,
    processed,
    ra_key_default,
    dec_key_default,
    id_key_default,
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
    batch_size,
    obj_batch_size,
    size,
    workers,
    update,
    show_stats,
    w_unions_cats,
    dl_tiles,
    build_kdtree,
    coordinates=None,
    dataframe_path=None,
    ra_key=None,
    dec_key=None,
    id_key=None,
    show_plt=False,
    save_plt=False,
):
    # check if the coordinates are provided as a list of pairs of coordinates or as a DataFrame
    if coordinates is not None:
        coordinates = coordinates[0]
        if (len(coordinates) == 0) or len(coordinates) % 2 != 0:
            raise ValueError('Provide even number of coordinates.')
        ras, decs, ids = (
            coordinates[::2],
            coordinates[1::2],
            list(np.arange(1, len(coordinates) // 2 + 1)),
        )
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        df_coordinates = pd.DataFrame({id_key: ids, ra_key: ras, dec_key: decs})

        formatted_coordinates = ' '.join([f'({ra}, {dec})' for ra, dec in zip(ras, decs)])
        logging.info(f'Coordinates received from the command line: {formatted_coordinates}')

        catalog = df_coordinates
        catalog_name = 'coordinates'
        coord_c = SkyCoord(
            catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs'
        )
    elif dataframe_path is not None:
        logging.info('Dataframe received from command line.')
        catalog = pd.read_csv(dataframe_path)
        catalog_name = os.path.basename(dataframe_path).split('.')[0]
        # if no ra_key, dec_key, id_key are provided, use the default ones
        if ra_key is None or dec_key is None or id_key is None:
            ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        # check if the keys are in the DataFrame
        if (
            ra_key not in catalog.columns
            or dec_key not in catalog.columns
            or id_key not in catalog.columns
        ):
            logging.error(
                'One or more keys not found in the DataFrame. Please provide the correct keys '
                'for right ascention, declination and object ID \n'
                'if they are not equal to the default keys: ra, dec, ID.'
            )
            return
        coord_c = SkyCoord(
            catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs'
        )
    elif w_unions_cats:
        logging.info('Using UNIONS catalogs.')
        catalog = pd.DataFrame()
        catalog_name = ''
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        coord_c = SkyCoord(
            catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs'
        )
    else:
        logging.info(
            'No coordinates or DataFrame provided. Using coordinates from default DataFrame.'
        )
        catalog = cat_default
        catalog_name = os.path.basename(cat_default).split('.')[0]
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        coord_c = SkyCoord(
            catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs'
        )
    # update information on the currently available tiles
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
    if w_unions_cats:
        unique_tiles, tiles_x_bands = tiles_from_unions_catalogs(
            availability, unions_det_dir, band_constr
        )
    else:
        # find the tiles the objects are in and check how many meet the band constraint
        unique_tiles, tiles_x_bands, catalog = tile_finder(
            availability, catalog, coord_c, tile_info_dir, band_constr
        )
    # log the number of unique tiles dwarfs are in
    logging.info(f'Number of tiles with detected objects: {len(unique_tiles)}')
    # log the number of catalog entries with a tile number
    logging.info(
        f'Total number of objects in the footprint: {len(catalog.loc[catalog.tile.notnull()])}/{len(catalog)}'
    )
    # log the number of tiles that meet the band constraint
    logging.info(
        f'Number of tiles with detected objects that meet the band constraint: {len(tiles_x_bands)}/{len(unique_tiles)}'
    )
    # log the number of dwarfs in the footprint that meet the band constraint
    logging.info(
        f'Number of detected objects in the footprint that meet the band constraint: '
        f'{len(catalog.loc[catalog.n_bands >= band_constr])}/{len(catalog)}'
    )
    # log the number of dwarfs available in 5, 4, 3, 2, 1 bands
    logging.info(
        f'Number of objects in the footprint that are available in\n5 bands: {len(catalog.loc[catalog.n_bands == 5])}\n4 bands: {len(catalog.loc[catalog.n_bands == 4])}\n3 bands: {len(catalog.loc[catalog.n_bands == 3])}\n2 bands: {len(catalog.loc[catalog.n_bands == 2])}\n1 band: {len(catalog.loc[catalog.n_bands == 1])}'
    )
    if 'cutout' not in catalog.columns:
        # initialize a column of zeros for the cutout column
        catalog['cutout'] = 0

    if print_per_tile_availability:
        # log information on the tile availability
        for tile in unique_tiles:
            bands = availability.get_availability(tile)[0]
            logging.info(f'Tile {tile} is available in {len(bands)} bands: {bands}')

    # process the tiles in batches
    total_batches = len(tiles_x_bands) // batch_size + (
        1 if len(tiles_x_bands) % batch_size != 0 else 0
    )
    for tile_idx, tile_batch in enumerate(
        process_tiles_in_batches(tiles_x_bands, batch_size), start=1
    ):
        logging.info(f'Processing batch: {tile_idx}/{total_batches}')
        start_batch_time = time.time()

        # download the tiles
        if dl_tiles:
            logging.info('Downloading the tiles in the available bands..')
            for tile in tile_batch:
                # check if there is already a .h5 file with tile in its name
                tile_pattern = re.compile(rf'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}')
                set_processed = read_processed(processed)
                if any(tile_pattern.search(file_name) for file_name in set_processed):
                    # if (
                    #     len(
                    #         glob.glob(
                    #             os.path.join(
                    #                 cutout_dir, f'*{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}*.h5'
                    #             )
                    #         )
                    #     )
                    #     != 0
                    # ):
                    logging.info(f'Tile {tile} already processed.')
                    continue

                start_download = time.time()
                if download_tile_for_bands_parallel(availability, tile, in_dict, download_dir):
                    logging.info(
                        f'Tile downloaded in all available bands. Took {np.round((time.time() - start_download) / 60, 3)} minutes.'
                    )
                else:
                    logging.info(f'Tile {tile} failed to download.')

        # log tile processing
        successful_tiles_count = 0
        total_cutouts_count = 0
        failed_tiles = []
        # initialize the result variables
        result = 0, 0, 0, True

        # process the tiles in parallel
        with ProcessPoolExecutor() as executor:
            future_to_tile = {
                executor.submit(
                    process_tile,
                    availability,
                    tile,
                    catalog,
                    dwarf_cat,
                    z_class_cat,
                    lens_cat,
                    processed,
                    id_key,
                    ra_key,
                    dec_key,
                    cutout_dir,
                    download_dir,
                    table_dir,
                    unions_det_dir,
                    in_dict,
                    size,
                    w_unions_cats,
                    obj_batch_size,
                ): tile
                for tile in tile_batch
            }

            for future in concurrent.futures.as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    result = future.result()
                    if (result[0] + result[1]) == result[2]:
                        logging.info('All objects in the tile were cut out.')
                        new_cutouts = result[0] + result[1]
                        total_cutouts_count += new_cutouts
                        successful_tiles_count += 1
                        catalog.loc[catalog['tile'] == tile, 'cutout'] = 1
                    else:
                        logging.error(
                            f'Something went wrong in tile {tile}! Only {result[0]+result[1]}/{result[2]} objects were cut out.'
                        )

                except Exception as e:
                    logging.exception(f'Failed to process tile {tile}: {str(e)}')
                    failed_tiles.append(tile)

        # skip updating master catalog if all tile batches have already been processed
        if w_unions_cats and not result[3]:
            # update the master catalog
            update_master_cat(cat_master, table_dir, tile_batch)

        if len(failed_tiles) == 0:
            logging.info('Tile batch processed sucessfully, deleting raw data.')
            for tile in tile_batch:
                tile_folder = os.path.join(
                    download_dir, f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
                )
                if os.path.exists(tile_folder):
                    shutil.rmtree(tile_folder)
        else:
            logging.info('Tile batch processed with errors, keeping raw data.')

        logging.info(
            f'\nProcessing report:\nTiles processed: {len(tile_batch)}\nCutouts created: {total_cutouts_count}'
            f'\nTiles failed: {len(failed_tiles)}/{len(tile_batch)}'
        )
        if len(failed_tiles) != 0:
            logging.info(f'Processing error in tiles: {failed_tiles}.')

        logging.info(
            f'Batch processing time: {np.round((time.time() - start_batch_time) / 60, 2)} minutes.'
        )

        # save the catalog with the suffix 'processed'
        if not catalog.empty:
            catalog.to_csv(
                os.path.join(table_dir, catalog_name + '_processed.csv'),
                index=False,
            )
        # plot all cutouts or just a random one
        if with_plot:
            if plot_random_cutout:
                random_tile_index = np.random.randint(0, len(tile_batch))
                avail_bands = ''.join(
                    availability.get_availability(tile_batch[random_tile_index])[0]
                )

                file_name_pattern = f'{str(tile_batch[random_tile_index][0]).zfill(3)}_{str(tile_batch[random_tile_index][1]).zfill(3)}_{size}x{size}_{avail_bands}'

                num_files_for_tile = len(
                    glob.glob(os.path.join(cutout_dir, f'*{file_name_pattern}*.h5'))
                )
                random_part_of_tile = np.random.randint(1, num_files_for_tile + 1)

                logging.info(
                    f'Plotting cutouts in random tile: {tile_batch[random_tile_index]} from the current batch.'
                )
                if w_unions_cats:
                    # open a random portion of the cutouts for the random tile
                    cutout_path = os.path.join(
                        cutout_dir,
                        file_name_pattern + f'_batch_{random_part_of_tile}.h5',
                    )
                else:
                    cutout_path = os.path.join(
                        cutout_dir,
                        file_name_pattern + '.h5',
                    )
                cutout = read_h5(cutout_path)
                random_obj_index = np.random.randint(0, cutout['images'].shape[0])
                # plot a random object from the stack of cutouts
                plot_cutout(
                    cutout,
                    in_dict,
                    figure_dir,
                    random_obj_index,
                    show_plot=show_plt,
                    save_plot=save_plt,
                )
            else:
                for idx in range(len(tile_batch)):
                    avail_bands = ''.join(availability.get_availability(tile_batch[idx])[0])
                    cutout_path = os.path.join(
                        cutout_dir,
                        f'{str(tile_batch[idx][0]).zfill(3)}_{str(tile_batch[idx][1]).zfill(3)}_{size}x{size}_{avail_bands}.h5',
                    )
                    cutout = read_h5(cutout_path)

                    plot_cutout(cutout, in_dict, figure_dir, show_plot=show_plt, save_plot=save_plt)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        action='append',
        metavar=('ra', 'dec'),
        help='list of pairs of coordinates to make cutouts from',
    )
    parser.add_argument('--dataframe', type=str, help='path to a CSV file containing the DataFrame')
    parser.add_argument('--ra_key', type=str, help='right ascension key in the DataFrame')
    parser.add_argument('--dec_key', type=str, help='declination key in the DataFrame')
    parser.add_argument('--id_key', type=str, help='id key in the DataFrame')
    args = parser.parse_args()

    # define the arguments for the main function
    arg_dict_main = {
        'cat_default': catalog_script,
        'dwarf_cat': dwarf_catalog,
        'z_class_cat': redshift_class_catalog,
        'lens_cat': lens_catalog,
        'cat_master': catalog_master,
        'processed': processed_file,
        'ra_key_default': ra_key_script,
        'dec_key_default': dec_key_script,
        'id_key_default': id_key_script,
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
        'batch_size': tile_batch_size,
        'obj_batch_size': object_batch_size,
        'size': cutout_size,
        'workers': num_workers,
        'update': update_tiles,
        'show_stats': show_tile_statistics,
        'w_unions_cats': with_unions_catalogs,
        'dl_tiles': download_tiles,
        'build_kdtree': build_new_kdtree,
        'coordinates': args.coordinates,
        'dataframe_path': args.dataframe,
        'ra_key': args.ra_key,
        'dec_key': args.dec_key,
        'id_key': args.id_key,
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

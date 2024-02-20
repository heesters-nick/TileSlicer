import logging
import os
import queue
import shutil
import threading  # Or multiprocessing, if preferred
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy.io import fits
from torch.utils.data import IterableDataset
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
# log_file_name = 'datastream.log'
# log_file_path = os.path.join(log_dir, log_file_name)

# # Configure the logging module
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file_path, mode='w'),
#         logging.StreamHandler(),  # Add this line to also log to the console
#     ],
# )

### tile parameters ###
band_constraint = 5  # define the minimum number of bands that should be available for a tile
cutout_size = 224  # square cutout size in pixels
num_workers = 5  # specifiy the number of threads for cutout creation
number_objects = 30000


# @nb.njit(nb.float32[:, :](nb.float32[:, :], nb.int32, nb.int32, nb.int32, nb.float32[:, :]))
def cutout2d(data_, x, y, size, cutout_in):
    y_large, x_large = data_.shape
    size_half = size // 2

    y_start = max(0, y - size_half)
    y_end = min(y_large, y + (size - size_half))

    x_start = max(0, x - size_half)
    x_end = min(x_large, x + (size - size_half))

    if y_start >= y_end or x_start >= x_end:
        raise ValueError('No overlap between the small and large array.')

    cutout_in[
        y_start - y + size_half : y_end - y + size_half,
        x_start - x + size_half : x_end - x + size_half,
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
        with fits.open(
            os.path.join(tile_dir, tile_fitsfilename), memmap=True, mode='readonly'
        ) as hdul:
            data = hdul[fits_ext].data.astype(np.float32)  # type: ignore
            logging.info(f'Opened {tile_fitsfilename}. Took {np.round(time.time()-fits_start, 2)}')
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
    except FileNotFoundError:
        logging.info(f'File {tile_fitsfilename} not found.')
        return None

    return cutouts_for_band


def cutout_all_bands(tile, in_dict, download_dir, obj_in_tile, size, workers):
    n_bands = len(in_dict)
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as executor:
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

    return final_cutouts


def cutout_all_bands_mod(
    tile,
    in_dict,
    download_dir,
    obj_in_tile,
    size,
    workers,
):
    n_bands = len(in_dict)
    slow_files = ['whigs-g', 'wishes-z']
    final_cutouts = np.zeros((len(obj_in_tile), n_bands, size, size), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=4) as executor_fast, ThreadPoolExecutor(
        max_workers=2
    ) as executor_slow:
        # Dictionary mapping each future to the corresponding band
        future_to_band = {}
        for band in list(in_dict):
            if band in slow_files:
                executor = executor_slow
            else:
                executor = executor_fast
            future_to_band[
                executor.submit(
                    cutout_one_band, tile, obj_in_tile, download_dir, in_dict, size, band
                )
            ] = band

        for future in as_completed(future_to_band):
            band = future_to_band[future]
            band_idx = list(in_dict.keys()).index(band)
            try:
                result = future.result()
                if result is not None:
                    final_cutouts[:, band_idx] = result
            except Exception as e:
                logging.exception(f'Failed to process band {band} for tile {tile}: {str(e)}')

    return final_cutouts


class DataStream(IterableDataset):
    def __init__(
        self,
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
        self.tile_info_dir = tile_info_dir
        self.unions_det_dir = unions_det_dir
        self.band_constr = band_constr
        self.download_dir = download_dir
        self.in_dict = in_dict
        self.cutout_size = cutout_size
        self.at_least_key = at_least_key
        self.dwarf_cat = dwarf_cat
        self.z_class_cat = z_class_cat
        self.lens_cat = lens_cat
        self.num_objects = num_objects
        self.show_stats = show_stats
        self.cutout_workers = cutout_workers
        self.queue_size = queue_size

        # maxsize determines how long the queue should be
        self.prefetch_queue = queue.Queue(maxsize=self.queue_size)
        self._initialize_tiles()

        # Initialize a lock for synchronizing access to the queue
        self.queue_lock = threading.Lock()

        # # Start the tile fetching and processing thread
        self.fetch_thread = threading.Thread(target=self._fetch_and_preprocess_tiles)
        # self.fetch_thread.start()

    def _initialize_tiles(self):
        logging.info('Initializing tiles..')
        # Extract available tile numbers using your existing functions
        u, g, lsb_r, i, z = extract_tile_numbers(load_available_tiles(self.tile_info_dir))
        all_bands = [u, g, lsb_r, i, z]
        self.availability = TileAvailability(all_bands, self.in_dict, self.at_least_key)

        # Optionally show tile statistics
        if self.show_stats:
            self.availability.stats()

        # Filter tiles based on detection catalogs and constraints
        _, self.tiles_x_bands = tiles_from_unions_catalogs(
            self.availability, self.unions_det_dir, self.band_constr
        )
        logging.info('Finished initializing tiles.')
        # Initialize tile index (for tracking which tile to process next)
        self.current_tile_index = 0

    def _determine_next_tile(self):
        if self.current_tile_index >= len(self.tiles_x_bands):
            return None  # Indicates no more tiles left

        tile_nums = self.tiles_x_bands[self.current_tile_index]
        self.current_tile_index += 1
        return tile_nums

    def _fetch_and_preprocess_tiles(self):
        logging.info('Fetching started.')
        self.fetch_start = time.time()
        while True:
            tile_info = self._determine_next_tile()
            if tile_info is None:
                break  # No more tiles left
            logging.info(f'Fetching tile {tuple(tile_info)}..')
            # Set up events and instance variables to monitor thread completion and success
            download_event = threading.Event()
            self.download_success = False
            catalog_event = threading.Event()
            self.catalog_success = False

            # Start tile download and pre-processing in a separate thread
            download_thread = threading.Thread(
                target=self._download_tile, args=(tile_info, download_event)
            )
            download_thread.start()

            # Process the object catalog concurrently
            catalog_thread = threading.Thread(
                target=self._process_catalog, args=(tile_info, catalog_event)
            )
            catalog_thread.start()

            # Wait for both download and catalog processing to finish
            download_event.wait()
            catalog_event.wait()

            if self.download_success and self.catalog_success:
                # Extract cutouts and feed stack and metadata into the queue
                self._extract_and_queue_cutouts(tile_info)
            else:
                # Error occured processing this tile, skip to the next
                continue

    def _download_tile(self, tile_nums, download_event, max_retries=3, retry_delay=5):
        download_start = time.time()
        retries = 0
        while retries < max_retries:
            try:
                # Download tile using your existing function
                if download_tile_for_bands_parallel(
                    self.availability, tile_nums, self.in_dict, self.download_dir
                ):
                    download_event.set()  # Signal that download is finished
                    self.download_success = True  # Download successful
                    logging.info(
                        f'Successfully downloaded tile {tile_nums} in {np.round(time.time()-download_start, 2)} seconds.'
                    )
                    return
            except Exception as e:
                logging.error(f'Error downloading tile {tile_nums}: {e}, attempt {retries+1}')

            retries += 1
            if retries < max_retries:
                logging.info(f'Retrying download for tile {tile_nums}...')
                time.sleep(retry_delay)  # Wait before retrying

        logging.error(f'Failed to download tile {tile_nums} after {max_retries} attempts.')
        self.download_success = False
        # Signal event is finished
        download_event.set()

    def _process_catalog(self, tile_nums, catalog_event):
        catalog_start = time.time()
        try:
            avail_bands = ''.join(self.availability.get_availability(tile_nums)[0])
            obj_in_tile = read_unions_cat(self.unions_det_dir, tile_nums)
            # add tile numbers to object dataframe
            obj_in_tile['tile'] = str(tile_nums)
            # add available bands to object dataframe
            obj_in_tile['bands'] = str(avail_bands)
            # add labels to the objects in the tile
            obj_in_tile = add_labels(
                obj_in_tile, self.dwarf_cat, self.z_class_cat, self.lens_cat, tile_nums
            )
            # only cutout part of the objects for testing
            obj_in_tile = obj_in_tile[: self.num_objects].reset_index(drop=True)
            self.processed_catalog = obj_in_tile
            self.catalog_success = True
        except Exception as e:
            logging.exception(f'Failed to process the object catalog for tile {tile_nums}: {e}.')
            self.catalog_success = False
        logging.info(f'Catalog processing took: {np.round(time.time()-catalog_start)} seconds.')
        # signal that catalog processing is finished
        catalog_event.set()

    def _extract_cutouts(self, obj_catalog, tile_nums):
        # len(obj_catalog) determines batch size
        cutout_start = time.time()
        try:
            cutout_stack = cutout_all_bands(
                tile_nums,
                self.in_dict,
                self.download_dir,
                obj_catalog,
                self.cutout_size,
                self.cutout_workers,
            )
            logging.info(
                f'Finished cutting in tile {tile_nums} in {np.round(time.time()-cutout_start, 2)} seconds.'
            )
            return cutout_stack
        except Exception as e:
            logging.exception(f'Something went wrong in cutout generation: {e}')
            return None

    def _extract_and_queue_cutouts(self, tile_nums):
        obj_catalog = self.processed_catalog  # Get processed object catalog
        cutout_batch = self._extract_cutouts(obj_catalog, tile_nums)

        # Acquire lock before accessing shared queue
        with self.queue_lock:
            self.prefetch_queue.put((cutout_batch, obj_catalog))

        logging.info(
            f'Fetch start to finished data product in {np.round(time.time()-self.fetch_start, 2)} seconds.'
        )

        tile_folder = os.path.join(
            self.download_dir, f'{str(tile_nums[0]).zfill(3)}_{str(tile_nums[1]).zfill(3)}'
        )
        if os.path.exists(tile_folder):
            logging.info(f'Cutting done, deleting tile {tile_nums}.')
            shutil.rmtree(tile_folder)

    def preload(self):
        self.fetch_thread.start()

        for _ in range(self.queue_size):
            self.prefetch_queue.get()  # Block and wait for each batch

        return True

    def __iter__(self):
        return self

    def __next__(self):
        logging.info('Pulling next tile.')
        with self.queue_lock:  # Thread lock the queue while accessing it
            start_wait_time = time.time()
            while self.prefetch_queue.empty():
                self.queue_lock.release()  # Release the lock while waiting
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting
                self.queue_lock.acquire()  # Reacquire the lock before checking the queue again

            wait_time = time.time() - start_wait_time
            logging.info(f'Waited for {wait_time:.2f} seconds for the next tile. Not good.')

            # Get the next data package once the queue is not empty
            return self.prefetch_queue.get()


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
    num_objects,
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

        # add labels to the objects in the tile
        obj_in_tile = add_labels(obj_in_tile, dwarf_cat, z_class_cat, lens_cat, tile)

        # only cutout part of the objects for testing
        obj_in_tile = obj_in_tile[:num_objects].reset_index(drop=True)

        cutting_start = time.time()
        cutout = cutout_all_bands(tile, in_dict, download_dir, obj_in_tile, size, workers)
        logging.info(f'Cutting finished. Took {np.round(time.time()-cutting_start, 2)} seconds.')
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

        tile_folder = os.path.join(download_dir, f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}')
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
        'num_objects': number_objects,
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

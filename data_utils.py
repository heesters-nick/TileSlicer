import glob
import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from filelock import FileLock
from tqdm import tqdm
from vos import Client

# client = Client()


def setup_logging(log_dir, script_name, logging_level):
    """
    Set up a custom logger for a given script

    Args:
        log_dir (str): directory where logs should be saved
        script_name (str): script name
    """
    log_filename = os.path.join(
        log_dir, f'{os.path.splitext(os.path.basename(script_name))[0]}.log'
    )

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()],
    )


def relate_coord_tile(coords=None, nums=None):
    """
    Conversion between tile numbers and coordinates.
    :param coords: right ascention, declination; tuple
    :param nums: first and second tile numbers; tuple
    :return: depending on the input, return the tile numbers or the ra and dec coordinates
    """
    if coords:
        ra, dec = coords
        xxx = ra * 2 * np.cos(np.radians(dec))
        yyy = (dec + 90) * 2
        return int(xxx), int(yyy)
    else:
        xxx, yyy = nums  # type: ignore
        dec = yyy / 2 - 90
        ra = xxx / 2 / np.cos(np.radians(dec))
        return np.round(ra, 12), np.round(dec, 12)


def tile_coordinates(name):
    """
    Extract RA and Dec from tile name
    :param name: .fits file name of a given tile
    :return RA and Dec of the tile center
    """
    parts = name.split('.')
    if name.startswith('calexp'):
        parts = parts[0].split('_')
    xxx, yyy = map(int, parts[1:3])
    ra = np.round(xxx / 2 / np.cos(np.radians((yyy / 2) - 90)), 6)
    dec = np.round((yyy / 2) - 90, 6)
    return ra, dec


def update_available_tiles(path, in_dict, save=True):
    """
    Update available tile lists from the VOSpace. Takes a few mins to run.

    Args:
        path (str): path to save tile lists.
        in_dict (dict): band dictionary
        save (bool): save new lists to disk, default is True.

    Returns:
        None
    """

    for band in np.array(list(in_dict.keys())):
        vos_dir = in_dict[band]['vos']
        band_filter = in_dict[band]['band']
        suffix = in_dict[band]['suffix']

        start_fetch = time.time()
        try:
            logging.info(f'Retrieving {band_filter}-band tiles...')
            band_tiles = Client().glob1(vos_dir, f'*{suffix}')
            end_fetch = time.time()
            logging.info(
                f'Retrieving {band_filter}-band tiles completed. Took {np.round((end_fetch-start_fetch)/60, 3)} minutes.'
            )
            if save:
                np.savetxt(os.path.join(path, f'{band}_tiles.txt'), band_tiles, fmt='%s')
        except Exception as e:
            logging.error(f'Error fetching {band_filter}-band tiles: {e}')


def load_available_tiles(path, in_dict):
    """
    Load tile lists from disk.
    Args:
        path (str): path to files
        in_dict (dict): band dictionary

    Returns:
        dictionary of available tiles for the selected bands
    """

    band_tiles = {}
    for band in np.array(list(in_dict.keys())):
        tiles = np.loadtxt(os.path.join(path, f'{band}_tiles.txt'), dtype=str)
        band_tiles[band] = tiles

    return band_tiles


def get_tile_numbers(name):
    """
    Extract tile numbers from tile name
    :param name: .fits file name of a given tile
    :return two three digit tile numbers
    """
    # parts = name.split('.')
    # if name.startswith('calexp'):
    #     parts = parts[0].split('_')
    # xxx, yyy = map(int, parts[1:3])

    if name.startswith('calexp'):
        pattern = re.compile(r'(?<=[_-])(\d+)(?=[_.])')
    else:
        pattern = re.compile(r'(?<=\.)(\d+)(?=\.)')

    matches = pattern.findall(name)

    return tuple(map(int, matches))


def extract_tile_numbers(tile_dict, in_dict):
    """
    Extract tile numbers from .fits file names.

    Args:
        tile_dict: lists of file names from the different bands
        in_dict: band dictionary

    Returns:
        num_lists (list): list of lists containing available tile numbers in the different bands
    """

    num_lists = []
    for band in np.array(list(in_dict.keys())):
        num_lists.append(np.array([get_tile_numbers(name) for name in tile_dict[band]]))

    return num_lists


class TileAvailability:
    def __init__(self, tile_nums, in_dict, at_least=False, band=None):
        self.all_tiles = tile_nums
        self.tile_num_sets = [set(map(tuple, tile_array)) for tile_array in self.all_tiles]
        self.unique_tiles = sorted(set.union(*self.tile_num_sets))
        self.availability_matrix = self._create_availability_matrix()
        self.counts = self._calculate_counts(at_least)
        self.band_dict = in_dict

    def _create_availability_matrix(self):
        array_shape = (len(self.unique_tiles), len(self.all_tiles))
        availability_matrix = np.zeros(array_shape, dtype=int)

        for i, tile in enumerate(self.unique_tiles):
            for j, tile_num_set in enumerate(self.tile_num_sets):
                availability_matrix[i, j] = int(tile in tile_num_set)

        return availability_matrix

    def _calculate_counts(self, at_least):
        counts = np.sum(self.availability_matrix, axis=1)
        bands_available, tile_counts = np.unique(counts, return_counts=True)

        counts_dict = dict(zip(bands_available, tile_counts))

        if at_least:
            at_least_counts = np.zeros_like(bands_available)
            for i, count in enumerate(bands_available):
                at_least_counts[i] = np.sum(tile_counts[i:])
            counts_dict = dict(zip(bands_available, at_least_counts))

        return counts_dict

    def get_availability(self, tile_nums):
        try:
            index = self.unique_tiles.index(tuple(tile_nums))
        except ValueError:
            logging.warning(f'Tile number {tile_nums} not available in any band.')
            return [], []
        except TypeError:
            return [], []
        bands_available = np.where(self.availability_matrix[index] == 1)[0]
        return [
            self.band_dict[list(self.band_dict.keys())[i]]['band'] for i in bands_available
        ], bands_available

    def band_tiles(self, band=None):
        return np.array(self.unique_tiles)[
            self.availability_matrix[:, list(self.band_dict.keys()).index(band)] == 1
        ]

    def stats(self, band=None):
        print('\nNumber of currently available tiles per band:\n')
        max_band_name_length = max(map(len, self.band_dict.keys()))  # for output format
        for band_name, count in zip(
            self.band_dict.keys(), np.sum(self.availability_matrix, axis=0)
        ):
            print(f'{band_name.ljust(max_band_name_length)}: \t {count}')

        print('\nNumber of tiles available in different bands:\n')
        for bands_available, count in sorted(self.counts.items(), reverse=True):
            print(f'In {bands_available} bands: {count}')

        print(f'\nNumber of unique tiles available:\n\n{len(self.unique_tiles)}')

        if band:
            print(f'\nNumber of tiles available in combinations containing the {band}-band:\n')

            all_bands = list(self.band_dict.keys())
            all_combinations = []
            for r in range(1, len(all_bands) + 1):
                all_combinations.extend(combinations(all_bands, r))
            combinations_w_r = [x for x in all_combinations if band in x]

            for band_combination in combinations_w_r:
                band_combination_str = ''.join([str(x).split('-')[-1] for x in band_combination])
                band_indices = [
                    list(self.band_dict.keys()).index(band_c) for band_c in band_combination
                ]
                common_tiles = np.sum(self.availability_matrix[:, band_indices].all(axis=1))
                print(f'{band_combination_str}: {common_tiles}')
            print('\n')


def read_h5(cutout_dir):
    """
    Reads cutout data from HDF5 file
    :param cutout_dir: cutout directory
    :return: cutout data
    """
    with h5py.File(cutout_dir, 'r') as f:
        # Create empty dictionaries to store data for each group
        cutout_data = {}

        # Loop through datasets
        for dataset_name in f:
            data = np.array(f[dataset_name])
            cutout_data[dataset_name] = data
    return cutout_data


def get_numbers_from_folders(unions_table_dir):
    """
    List UNIONS directories and extract tile numbers.

    Args:
        unions_table_dir (str): directory where the UNIONS catalogs are located

    Returns:
        list: tile list
    """
    tile_list = []

    # Iterate through each folder in the specified path
    for folder_name in os.listdir(unions_table_dir):
        folder_full_path = os.path.join(unions_table_dir, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_full_path):
            try:
                numbers_tuple = tuple(map(int, folder_name.split('.')[1:3]))
                # not all folders contain detection catalogs
                if os.path.exists(
                    os.path.join(folder_full_path, folder_name + '_ugri_photoz_ext.cat')
                ) and numbers_tuple not in [(86, 337)]:
                    tile_list.append(numbers_tuple)
            except ValueError:
                # Handle the case where the folder name doesn't match the expected format
                logging.warning(
                    f"Skipping folder {folder_name} as it doesn't match the expected format."
                )

    return tile_list


def read_unions_cat(unions_table_dir, tile_nums):
    """
    Read UNIONS catalog from disk.

    Args:
        unions_table_dir (str): directory where the UNIONS catalogs are located
        tile_nums (list): list of tile numbers

    Returns:
        cat (dataframe): pandas dataframe containing the UNIONS catalog for the specified tile
    """
    logging.debug(f'Reading UNIONS catalog for tile {tile_nums}')
    try:
        df = Table.read(
            os.path.join(
                unions_table_dir,
                f'UNIONS.{str(tile_nums[0]).zfill(3)}.{str(tile_nums[1]).zfill(3)}',
                f'UNIONS.{str(tile_nums[0]).zfill(3)}.{str(tile_nums[1]).zfill(3)}_ugri_photoz_ext.cat',
            ),
            hdu=1,
        ).to_pandas()
        columns = ['SeqNr', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'MAG_GAAP_r']
        df = df[columns].rename(
            columns={
                'SeqNr': 'ID',
                'X_IMAGE': 'x',
                'Y_IMAGE': 'y',
                'ALPHA_J2000': 'ra',
                'DELTA_J2000': 'dec',
                'MAG_GAAP_r': 'mag_r',
            }
        )
        # replace -99 and +99 values with NaN
        df['mag_r'].replace([-99.0, 99.0], np.nan, inplace=True)

        logging.debug(f'Read {len(df)} objects from UNIONS catalog for tile {tile_nums}')
    except PermissionError:
        logging.error(f'Permission error reading UNIONS catalog for tile {tile_nums}')
        df = None
    return df


def read_dwarf_cat(dwarf_cat, tile_nums):
    """
    Read known dwarfs in tile to dataframe.

    Args:
        dwarf_cat (str): path to dwarf catalog
        tile_nums (tuple): tile numbers

    Returns:
        cat (dataframe): pandas dataframe containing the dwarf catalog for the specified tile
    """
    logging.debug(f'Reading dwarf catalog for tile {tile_nums}')
    try:
        df = pd.read_csv(dwarf_cat)
        df = df[df['tile'] == str(tuple(tile_nums))].reset_index(drop=True)
    except FileNotFoundError:
        logging.error(f'File not found: {dwarf_cat}')
        df = None
    return df


def match_cats(df_det, df_label, max_sep=15.0):
    """
    Match detections to known objects, return matches, unmatches

    Args:
        df_det (dataframe): detections dataframe
        df_label (dataframe): dataframe of objects with labels
        max_sep (float): maximum separation tollerance

    Returns:
        det_matching_idx (list): indices of detections for which labels are available
        label_matches (dataframe): known objects that were detected
        det_matches (dataframe): detections that are known objects
    """
    c_det = SkyCoord(df_det['ra'], df_det['dec'], unit=u.deg)
    c_label = SkyCoord(df_label['ra'], df_label['dec'], unit=u.deg)

    idx, d2d, _ = c_label.match_to_catalog_3d(c_det)
    # sep_constraint is a list of True/False
    sep_constraint = d2d < max_sep * u.arcsec
    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matching_idx = idx[sep_constraint]  # det_matching_idx is a list of indices
    det_matches = df_det.loc[det_matching_idx].reset_index(drop=True)
    return det_matching_idx, label_matches, label_unmatches, det_matches


def read_parquet(parquet_path, ra_range, dec_range, columns=None):
    """
    Read parquet file and return a pandas dataframe.

    Args:
        parquet_path (str): path to parquet file
        ra_range (tuple): range of right ascension to select
        dec_range (tuple): range of declination to select
        columns (list): columns to select

    Returns:
        df (dataframe): pandas dataframe containing the selected data
    """
    logging.debug('Reading redshift catalog.')
    filter_coords = [
        ('ra', '>=', ra_range[0]),
        ('ra', '<=', ra_range[1]),
        ('dec', '>=', dec_range[0]),
        ('dec', '<=', dec_range[1]),
    ]
    df = pq.read_table(parquet_path, memory_map=True, filters=filter_coords).to_pandas()
    if columns:
        df = df[columns]
    logging.debug(f'Read {len(df)} objects from catalog {os.path.basename(parquet_path)}.')
    return df


def add_labels(det_df, dwarf_cat, z_class_cat, lens_cat, tile_nums):
    """
    Add labels to detections dataframe.

    Args:
        det_df (dataframe): detections dataframe
        dwarf_cat (str): path to dwarf catalog
        z_class_cat (str): path to redshift and class catalog
        lens_cat (str): path to lens catalog
        tile_nums (tuple): tile numbers

    Returns:
        det_df (dataframe): detections dataframe with labels
    """
    det_df = det_df.copy()

    logging.debug(f'Adding labels to the detections dataframe for tile {tile_nums}.')
    # define minimum and maximum ra and dec values to filter the label catalog
    margin = 0.0014  # extend the ra and dec ranges by this amount in degrees
    ra_range = (np.min(det_df['ra']) - margin, np.max(det_df['ra']) + margin)
    dec_range = (np.min(det_df['dec'] - margin), np.max(det_df['dec'] + margin))

    # read the redshift/class catalog
    class_z_df = read_parquet(
        z_class_cat,
        ra_range=ra_range,
        dec_range=dec_range,
    )
    # match detections to redshift and class catalog
    det_idx_z_class, label_matches_z_class, _, _ = match_cats(det_df, class_z_df, max_sep=1.0)
    det_idx_z_class = det_idx_z_class.astype(np.int32)  # make sure index is int

    # add redshift and class labels to detections dataframe
    det_df.loc[:, 'class_label'] = np.nan
    det_df.loc[:, 'zspec'] = np.nan
    det_df.loc[det_idx_z_class, 'class_label'] = label_matches_z_class['cspec'].values
    det_df.loc[det_idx_z_class, 'zspec'] = label_matches_z_class['zspec'].values

    logging.debug(
        f'Number of detection matches for z/class: {len(det_idx_z_class)}, number of label matches for z/class: {len(label_matches_z_class)} for tile {det_df["tile"].iloc[0]}'
    )

    logging.debug(
        f'Found {np.count_nonzero(~np.isnan(label_matches_z_class["zspec"]))} matching objects in the redshift/class catalog for tile {det_df["tile"].iloc[0]}.'
    )

    logging.debug(
        f'Added {np.count_nonzero(~np.isnan(det_df["zspec"]))} redshift labels to the detection dataframe for tile {det_df["tile"].iloc[0]}.'
    )
    logging.debug(
        f'Added {np.count_nonzero(~np.isnan(det_df["class_label"]))} class labels to the detection dataframe for tile {det_df["tile"].iloc[0]}.'
    )

    # read the lens catalog
    lens_df = read_parquet(lens_cat, ra_range=ra_range, dec_range=dec_range)
    # match detections to lens catalog
    det_idx_lens, label_matches_lens, _, _ = match_cats(det_df, lens_df, max_sep=1.0)
    det_idx_lens = det_idx_lens.astype(np.int32)  # make sure index is int
    # add lens labels to detections dataframe
    det_df.loc[:, 'lens'] = np.nan
    det_df.loc[det_idx_lens, 'lens'] = 1

    logging.debug(
        f'Number of detection matches for lenses: {len(det_idx_lens)}, number of label matches for lenses: {len(label_matches_lens)} for tile {det_df["tile"].iloc[0]}.'
    )

    logging.debug(
        f'Found {len(label_matches_lens)} matching objects in the lens catalog for tile {det_df["tile"].iloc[0]}.'
    )

    logging.debug(
        f'Added {np.count_nonzero(~np.isnan(det_df["lens"]))} lens labels to the detection dataframe for tile {det_df["tile"].iloc[0]}.'
    )

    dwarfs_df = read_dwarf_cat(dwarf_cat, tile_nums)
    # match detections to dwarf catalog
    det_idx_lsb, _, lsb_unmatches, _ = match_cats(det_df, dwarfs_df, max_sep=10.0)
    # add lsb labels to detections dataframe
    det_df.loc[:, 'lsb'] = np.nan
    det_df.loc[det_idx_lsb, 'lsb'] = 1
    det_df.loc[det_idx_lsb, 'class_label'] = 2  # dwarfs are galaxies

    logging.debug(
        f'Added {np.count_nonzero(~np.isnan(det_df["lsb"]))} LSB labels to the detection dataframe for tile {det_df["tile"].iloc[0]}.'
    )

    if len(lsb_unmatches) > 0:
        logging.debug(
            f'Found {len(lsb_unmatches)} undetected but known dwarfs in tile {dwarfs_df.tile[0]}.'  # type: ignore
        )
        lsb_unmatches.loc[:, 'lsb'] = 1  # dwarfs are LSB
        lsb_unmatches.loc[:, 'class_label'] = 2  # dwarfs are galaxies
        # augment detections dataframe with undetected but known dwarfs
        common_columns = det_df.columns.intersection(lsb_unmatches.columns)
        det_df = pd.concat([det_df, lsb_unmatches[common_columns]], ignore_index=True)
    logging.debug('Finished adding labels to the detections dataframe.')
    return det_df


def save_tile_cat(table_dir, tile_nums, obj_in_tile):
    """
    Save the tile catalog to a temporary file.

    Args:
        table_dir (str): path to directory where catalogs are stored
        tile_nums (tuple): tile numbers
        obj_in_tile (dataframe): objects that were cut out in the current tile
    """
    logging.debug(f'Saving tile catalog for tile {tile_nums} to a temporary file.')
    temp_path = os.path.join(table_dir, f'cat_temp_{tile_nums[0]}_{tile_nums[1]}.parquet')
    columns = [
        'ra',
        'dec',
        'cfis_id',
        'class_label',
        'lens',
        'lsb',
        'mag_r',
        'tile',
        'zspec',
        'bands',
    ]
    obj_in_tile[columns].to_parquet(temp_path, index=False)


def update_master_cat(cat_master, table_dir, batch_tile_list):
    """
    Fuse catalogs from tiles in the batch and append the fused catalog to the master catalog.

    Args:
        cat_master (str): path to master catalog
        table_dir (str): path to directory where catalogs are stored
        batch_tile_list (list): list of tile numbers in the batch
    """
    logging.info('Updating the master catalog..')
    tile_cats = []
    for tile in batch_tile_list:
        temp_path = os.path.join(table_dir, f'cat_temp_{tile[0]}_{tile[1]}.parquet')
        if os.path.exists(temp_path):
            tile_cat = pq.read_table(temp_path, memory_map=True).to_pandas()
            tile_cats.append(tile_cat)
            os.remove(temp_path)
    if len(tile_cats) == 0:
        logging.info('No catalogs to fuse.')
        return
    # fuse catalogs in the tile batch
    batch_tile_cats = pd.concat(tile_cats, ignore_index=True)

    # update master catalog if it exists
    if os.path.exists(cat_master):
        master_table = pq.read_table(cat_master, memory_map=True).to_pandas()
        master_table_updated = pd.concat([master_table, batch_tile_cats], ignore_index=True)

        # Check for duplicate rows
        duplicate_rows = master_table_updated[master_table_updated.duplicated()]

        if not duplicate_rows.empty:
            logging.debug('Duplicate rows found. Dropping them.')

            # Remove duplicate rows
            master_table_updated = master_table_updated.drop_duplicates()
        else:
            logging.debug('No duplicate rows found. Moving on.')

        master_table_updated.to_parquet(cat_master, index=False)
    # create a new master catalog if it does not exist yet
    else:
        batch_tile_cats.to_parquet(cat_master, index=False)
    logging.info('Updated the master catalog.')


def object_batch_generator(obj_df, batch_size):
    """
    Generate batches of objects from a dataframe.

    Args:
        obj_df (dataframe): dataframe containing objects
        batch_size (int): batch size

    Yields:
        dataframe: batch of objects
    """
    total_rows = len(obj_df)
    for i in range(0, total_rows, batch_size):
        yield obj_df.iloc[i : min(i + batch_size, total_rows)]


def is_array_non_empty(array):
    return any(element != 0 for element in array)


def get_filenames(directory_path, pattern='*.h5'):
    """
    Read filenames from a directory.

    Args:
        directory_path (str): path to directory
        pattern (str, optional): search pattern. Defaults to '*.h5'.

    Returns:
        list: filenames
    """
    try:
        filenames = [
            os.path.basename(file) for file in glob.glob(os.path.join(directory_path, pattern))
        ]
        return filenames
    except OSError as e:
        logging.error(f'Error reading directory: {e}')
        return []


def write_file_names_to_file(file_names, save_path):
    """
    Write file names to a file.

    Args:
        file_names (list): file names
        save_path (str): path to file
    """
    try:
        with open(save_path, 'w') as file:
            file.writelines(f'{name}\n' for name in file_names)
        logging.info(f'File names written to {save_path}')
    except IOError as e:
        logging.error(f'Error writing to file: {e}')


def read_processed(file_path):
    """
    Read processed file names from a file.

    Args:
        file_path (str): path to file

    Returns:
        set: set of file names
    """
    try:
        with FileLock(file_path + '.lock'):
            with open(file_path, 'r') as file:
                return {line.strip() for line in file.readlines()}
    except FileNotFoundError:
        logging.info(f'File {file_path} not found. No tiles processed yet, returning empty set.')
        return set()
    except IOError as e:
        logging.error(f'Error reading file: {e}')
        return set()


def update_processed(file_name, file_path):
    """
    Update processed file names in a file.

    Args:
        file_name (str): file name
        file_path (str): path to file
    """
    logging.info(f'Adding {file_name} to processed file.')
    try:
        with FileLock(file_path + '.lock'):
            # Create an empty file if it does not exist
            if not os.path.exists(file_path):
                open(file_path, 'a').close()
            with open(file_path, 'a') as file:
                file.write(file_name + '\n')
    except IOError as e:
        logging.error(f'Error writing to file: {e}')


def update_h5_labels_parallel(tile_nums, h5_path_list, unions_det_dir, z_class_cat, lens_cat):
    """
    Update the labels in the cutout files in the HDF5 file based on better matching.

    Args:
        tile_nums (tuple): tile numbers
        h5_path_list (str): list of HDF5 file paths for a given tile
        unions_det_dir (str): path to directory containing unions catalogs
        z_class_cat (dataframe): dataframe containing the redshift and classification
        lens_cat (dataframe): dataframe containing known lenses
    """

    # read detections for the tile into a dataframe
    obj_df = read_unions_cat(unions_det_dir, tile_nums)
    if obj_df is not None:
        margin = 0.0014  # extend the ra and dec ranges by this amount in degrees
        ra_range = (np.min(obj_df['ra']) - margin, np.max(obj_df['ra']) + margin)
        dec_range = (np.min(obj_df['dec'] - margin), np.max(obj_df['dec'] + margin))

        # read the label catalog
        class_z_df = read_parquet(
            z_class_cat,
            ra_range=ra_range,
            dec_range=dec_range,
        )
        # match detections to redshift and class catalog
        det_idx, label_matches, _, _ = match_cats(obj_df, class_z_df, max_sep=1.0)
        # add redshift and class labels to detections dataframe
        obj_df['class_label'] = np.nan
        obj_df['zspec'] = np.nan
        det_idx = det_idx.astype(np.int32)  # make sure index is int
        obj_df.loc[det_idx, 'class_label'] = label_matches['cspec'].values
        obj_df.loc[det_idx, 'zspec'] = label_matches['zspec'].values

        # read the lens catalog
        lens_df = read_parquet(lens_cat, ra_range=ra_range, dec_range=dec_range)
        # match detections to lens catalog
        det_idx_lens, label_matches_lens, _, _ = match_cats(obj_df, lens_df, max_sep=1.0)
        det_idx_lens = det_idx_lens.astype(np.int32)  # make sure index is int
        # add lens labels to detections dataframe
        obj_df['lens'] = np.nan
        obj_df.loc[det_idx_lens, 'lens'] = 1

        # change to float32 for matching
        obj_df['ra'] = obj_df['ra'].astype('float32')
        obj_df['dec'] = obj_df['dec'].astype('float32')
        obj_df['zspec'] = obj_df['zspec'].astype('float32')
        obj_df['class_label'] = obj_df['class_label'].astype('float32')
    else:
        logging.error(f'No UNIONS catalog for tile {tile_nums} found.')

    with ThreadPoolExecutor() as executor:
        executor.map(
            update_single_h5,
            h5_path_list,
            [obj_df] * len(h5_path_list),
        )


def update_single_h5(h5_path, obj_dataframe):
    """
    Update the labels in one HDF5 file.

    Args:
        h5_path (str): path to HDF5 file
        obj_dataframe (dataframe): detection dataframe
    """
    file_name = os.path.basename(h5_path)
    logging.info(f'Updating labels in {file_name}.')
    with h5py.File(h5_path, 'r+') as hf:
        try:
            ra = hf['ra']
            dec = hf['dec']

            # create pandas dataframe
            df = pd.DataFrame({'ra': ra, 'dec': dec})

            # get part of the obj_df where ra and dec are equal to ra/dec in df
            merged_df = pd.merge(df, obj_dataframe, on=['ra', 'dec'], how='inner')

            dataset_names = ['zspec', 'class_label', 'lens']  # Specify the dataset names
            # create empty dataframe
            h5_df = pd.DataFrame()
            # Check if datasets exist and create a DataFrame
            for dataset_name in dataset_names:
                try:
                    h5_df[dataset_name] = hf[dataset_name]
                except KeyError:
                    logging.warning(
                        f"Dataset '{dataset_name}' does not exist in the HDF5 file {file_name}."
                    )
                    logging.debug(f'Creating dataset {dataset_name} in the HDF5 file {file_name}.')
                    h5_df[dataset_name] = np.nan * np.ones(len(df))

            z_matching = np.count_nonzero(
                merged_df['zspec'].fillna('nan') == h5_df['zspec'].fillna('nan')
            )
            class_matching = np.count_nonzero(
                merged_df['class_label'].fillna('nan') == h5_df['class_label'].fillna('nan')
            )

            lens_matching = np.count_nonzero(
                merged_df['lens'].fillna('nan') == h5_df['lens'].fillna('nan')
            )

            logging.debug(
                f'Found {np.count_nonzero(~merged_df["zspec"].isna())} objects with available redshift labels for {file_name}.'
            )
            logging.debug(
                f'Found {np.count_nonzero(~merged_df["class_label"].isna())} objects with available class labels for {file_name}.'
            )
            logging.debug(
                f'Found {np.count_nonzero(~merged_df["lens"].isna())} objects that are known lenses for {file_name}.'
            )

            logging.debug(f'File {file_name}: changed redshifts: {len(df)-z_matching}/{len(df)}')
            logging.debug(f'File {file_name}: changed classes: {len(df)-class_matching}/{len(df)}')
            logging.debug(
                f'File {file_name}: changed lens labels: {len(df)-lens_matching}/ {len(df)}'
            )

            for dataset_name in dataset_names:
                if dataset_name in hf.keys():
                    hf[dataset_name][:] = merged_df[dataset_name].values  # type: ignore
                else:
                    hf.create_dataset(
                        dataset_name, data=merged_df[dataset_name].values.astype(np.float32)
                    )

            logging.info(f'Updated labels in {file_name}.')

        except KeyError:
            logging.error(f'Failed reading {file_name}.')


def download_decals_cutouts(ra, dec, pixscale, output_folder='downloads', concurrent_downloads=1):
    """
    Download cutouts from the Legacy Survey.

    Args:
        ra (numpy.ndarray): right ascention
        dec (numpy.ndarray): declination
        pixscale (float): pixel scale in arcseconds/pixel
        output_folder (str, optional): where to save the files. Defaults to 'downloads'.
        concurrent_downloads (int, optional): number of files to download in parallel. Defaults to 1.
    """
    base_url = 'https://www.legacysurvey.org/viewer/jpeg-cutout'

    # Ensure inputs are NumPy arrays
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    pixscale = np.atleast_1d(pixscale)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Function to download a single file
    def download_single(url, output_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            # print(f"Downloaded: {output_path}")
        except Exception as e:
            logging.error(f'Error downloading {url}: {e}')

    # Download files concurrently
    with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
        futures = []
        for i, (r, d, p) in enumerate(zip(ra, dec, pixscale)):
            url = f'{base_url}?ra={r}&dec={d}&layer=ls-dr10&pixscale={p}'
            output_path = os.path.join(output_folder, f'cutout_{i + 1}.jpg')
            futures.append(executor.submit(download_single, url, output_path))

        # Wait for all downloads to complete
        for future in tqdm(futures, desc='Downloading', unit='file'):
            future.result()


def create_master_cat_from_file(cat_master, table_dir, h5_path_list, tile_nums):
    """
    Generate a master catalog of all cut out objects from the hdf5 files.

    Args:
        cat_master (str): path to master catalog
        table_dir (str): table directory
        h5_path_list (list): list of paths to hdf5 files
        tile_nums (tuple): tile numbers
    """
    df_tile = pd.DataFrame()

    for h5_path in h5_path_list:
        with h5py.File(h5_path, 'r') as hf:
            # Extracting available tiles
            pattern = r'224x224_(.*?)_batch'
            match = re.search(pattern, h5_path)
            if match:
                av_bands = match.group(1)
            else:
                logging.error(f'The available bands could not be extracted from path {h5_path}')
                av_bands = ''

            # Initialize empty dictionary to store datasets
            datasets_dict = {}
            try:
                # Iterate over the keys (dataset names) in the HDF5 file
                for dataset_name in hf.keys():
                    if dataset_name == 'images':
                        continue
                    if not dataset_name == 'tile':
                        # Read the dataset into a NumPy array
                        data = hf[dataset_name][:]  # type: ignore
                    else:
                        # Repeat tile numbers in array
                        data = np.full(len(hf['ra'][:]), str(tuple(hf[dataset_name][:])))  # type: ignore

                    # Store the dataset in the dictionary with its name as the key
                    datasets_dict[dataset_name] = data

                datasets_dict['bands'] = np.full(len(hf['ra'][:]), av_bands)  # type: ignore
            except KeyError:
                logging.error(f'Failed reading {os.path.basename(h5_path)}.')

        # Create a Pandas DataFrame from the dictionary
        df = pd.DataFrame(datasets_dict)
        if not df.empty:
            df_order = ['ra', 'dec'] + [col for col in df.columns if col not in ['ra', 'dec']]
            df = df[df_order]
            df_tile = pd.concat([df_tile, df], ignore_index=True)
        else:
            logging.warning(f'No objects found in {os.path.basename(h5_path)}.')

    if not df_tile.empty:
        # update master catalog if it exists
        if os.path.exists(cat_master):
            master_table = pq.read_table(cat_master, memory_map=True).to_pandas()
            master_table_updated = pd.concat([master_table, df_tile], ignore_index=True)

            # Check for duplicate rows
            duplicate_rows = master_table_updated[master_table_updated.duplicated()]

            if not duplicate_rows.empty:
                logging.debug(f'Duplicate rows found in tile {tile_nums}. Dropping them.')

                # Remove duplicate rows
                master_table_updated = master_table_updated.drop_duplicates()
            else:
                logging.debug(f'No duplicate rows found in tile {tile_nums}. Moving on.')

            master_table_updated.to_parquet(cat_master, index=False)
        # create a new master catalog if it does not exist yet
        else:
            df_tile.to_parquet(cat_master, index=False)
    else:
        logging.warning(f'No objects found in tile {tile_nums}.')


def extract_numbers(file_name):
    """
    Extract the tile numbers from the file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        tuple: The tile numbers.
    """
    try:
        # extract tile numbers
        substring = file_name.split('_')[:2]
        # convert to integers
        return tuple(map(int, substring))
    except (ValueError, IndexError):
        return ()


def kahan_sum(nums):
    """
    Sum function to sum very large or very small numbers without losing precision.

    Args:
        nums (numpy.ndarray): array of numbers to be summed
    Returns:
        float: sum of numbers
    """
    sum = 0.0
    c = 0.0  # A compensation for low precision
    for num in nums:
        y = num - c  # Subtract the compensation
        t = sum + y  # Add num to sum
        c = (t - sum) - y  # Calculate new compensation
        sum = t
    return sum


def update_df_tile_stats(df_path, new_row):
    """
    Updates the dataframe used to accumulate tile statistics by appending a new row.

    Args:
        df_path (str): path to existing dataframe
        new_row (dataframe): new row to append
    """
    if os.path.isfile(df_path):
        new_row.to_csv(df_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(df_path, index=False)


def find_band_indices(data, bands):
    indices = []
    # Create a lookup for bands to their indices
    band_to_index = {value['band']: idx for idx, value in enumerate(data.values())}
    # Loop through the requested bands and get their indices using the lookup
    for band in bands:
        index = band_to_index.get(band, -1)  # Get index or -1 if not found
        indices.append(index)
    return sorted(indices)


def adjust_flux_with_zp(flux, current_zp, standard_zp):
    adjusted_flux = flux * 10 ** (-0.4 * (current_zp - standard_zp))
    return adjusted_flux


def find_and_save_file_names(directory, pattern, output_file):
    """
    Finds files matching a specific pattern in a directory and writes their filenames to a file.

    Args:
    directory (str): The directory to search in.
    pattern (str): The file pattern to search for.
    output_file (str): File to write the filenames to.
    """
    # Construct the find command
    find_command = f"find {directory} -type f -name '{pattern}' -print0"

    # Use basename with xargs to extract filenames
    xargs_command = 'xargs -0 -I {} basename {}'

    # Execute the find command and pipe to xargs, then redirect to output file
    command = f'{find_command} | {xargs_command} > {output_file}'
    subprocess.run(command, shell=True, check=True)


def list_vospace(directory, suffix=''):
    """Use vls to list contents of a VOSpace directory"""
    try:
        result = subprocess.run(
            ['vls', f'{directory}{suffix}'], text=True, capture_output=True, check=True
        )
        entry_list = [line.strip() for line in result.stdout.splitlines()]
        return entry_list
    except subprocess.CalledProcessError as e:
        logging.error(f'Error listing {directory}: {e}')
        return []


def filter_files(files, suffix):
    return [file for file in files if file.endswith(suffix)]

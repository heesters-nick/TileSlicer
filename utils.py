import glob
import logging
import os
import time

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from vos import Client

client = Client()


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


def update_available_tiles(path, save=True):
    """
    Update available tile lists from the VOSpace. Takes a few mins to run.

    Args:
        path (str): path to save tile lists.
        save (bool): save new lists to disk, default is True.

    Returns:
        None
    """
    logging.info('Updating available tile lists from the VOSpace.')
    logging.info('Retrieving u-band tiles...')
    start_u = time.time()
    cfis_u_tiles = client.glob1('vos:cfis/tiles_DR5/', '*u.fits')
    end_u = time.time()
    logging.info(
        f'Retrieving u-band tiles completed. Took {np.round((end_u-start_u)/60, 3)} minutes.'
    )
    logging.info('Retrieving g-band tiles...')
    whigs_g_tiles = client.glob1('vos:cfis/whigs/stack_images_CFIS_scheme/', '*.fits')
    end_g = time.time()
    logging.info(
        f'Retrieving g-band tiles completed. Took {np.round((end_g-end_u)/60, 3)} minutes.'
    )
    logging.info('Retrieving r-band tiles...')
    cfis_lsb_r_tiles = client.glob1('vos:cfis/tiles_LSB_DR5/', '*.fits')
    end_r = time.time()
    logging.info(
        f'Retrieving r-band tiles completed. Took {np.round((end_r-end_g)/60, 3)} minutes.'
    )
    logging.info('Retrieving i-band tiles...')
    ps_i_tiles = client.glob1('vos:cfis/panstarrs/DR3/tiles/', '*i.fits')
    end_i = time.time()
    logging.info(
        f'Retrieving i-band tiles completed. Took {np.round((end_i-end_r)/60, 3)} minutes.'
    )
    logging.info('Retrieving z-band tiles...')
    wishes_z_tiles = client.glob1('vos:cfis/wishes_1/coadd/', '*.fits')
    end_z = time.time()
    logging.info(
        f'Retrieving z-band tiles completed. Took {np.round((end_z-end_i)/60, 3)} minutes.'
    )
    if save:
        np.savetxt(path + 'cfis_u_tiles.txt', cfis_u_tiles, fmt='%s')
        np.savetxt(path + 'whigs_g_tiles.txt', whigs_g_tiles, fmt='%s')
        np.savetxt(path + 'cfis_lsb_r_tiles.txt', cfis_lsb_r_tiles, fmt='%s')
        np.savetxt(path + 'ps_i_tiles.txt', ps_i_tiles, fmt='%s')
        np.savetxt(path + 'wishes_z_tiles.txt', wishes_z_tiles, fmt='%s')


def load_available_tiles(path):
    """
    Load tile lists from disk.
    :param path: path to files
    :return: lists of available tiles for the five bands
    """
    u_tiles = np.loadtxt(path + 'cfis_u_tiles.txt', dtype=str)
    g_tiles = np.loadtxt(path + 'whigs_g_tiles.txt', dtype=str)
    lsb_r_tiles = np.loadtxt(path + 'cfis_lsb_r_tiles.txt', dtype=str)
    i_tiles = np.loadtxt(path + 'ps_i_tiles.txt', dtype=str)
    z_tiles = np.loadtxt(path + 'wishes_z_tiles.txt', dtype=str)

    return u_tiles, g_tiles, lsb_r_tiles, i_tiles, z_tiles


def get_tile_numbers(name):
    """
    Extract tile numbers from tile name
    :param name: .fits file name of a given tile
    :return two three digit tile numbers
    """
    parts = name.split('.')
    if name.startswith('calexp'):
        parts = parts[0].split('_')
    xxx, yyy = map(int, parts[1:3])
    return xxx, yyy


def extract_tile_numbers(tile_lists):
    """
    Extract tile numbers from .fits file names.
    :param tile_lists: lists of file names from the different bands
    :return: lists of tile numbers available in the different bands
    """
    u_nums = np.array([get_tile_numbers(name) for name in tile_lists[0]])
    g_nums = np.array([get_tile_numbers(name) for name in tile_lists[1]])
    lsb_r_nums = np.array([get_tile_numbers(name) for name in tile_lists[2]])
    i_nums = np.array([get_tile_numbers(name) for name in tile_lists[3]])
    z_nums = np.array([get_tile_numbers(name) for name in tile_lists[4]])

    return u_nums, g_nums, lsb_r_nums, i_nums, z_nums


class TileAvailability:
    def __init__(self, tile_nums, band_dict, at_least=False, band=None):
        self.all_tiles = tile_nums
        self.tile_num_sets = [set(map(tuple, tile_array)) for tile_array in self.all_tiles]
        self.unique_tiles = sorted(set.union(*self.tile_num_sets))
        self.availability_matrix = self._create_availability_matrix()
        self.counts = self._calculate_counts(at_least)
        self.band_dict = band_dict

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
            # print(f'Tile number {tile_number} not available in any band.')
            return [], []
        except TypeError:
            return [], []
        bands_available = np.where(self.availability_matrix[index] == 1)[0]
        return [
            self.band_dict[list(self.band_dict.keys())[i]]['band'] for i in bands_available
        ], bands_available

    def band_tiles(self, band):
        return np.array(self.unique_tiles)[
            self.availability_matrix[:, list(self.band_dict.keys()).index(band)] == 1
        ]

    def stats(self):
        print('\nNumber of currently available tiles per band:\n')
        max_band_name_length = max(map(len, self.band_dict.keys()))  # for output format
        for band_name, count in zip(
            self.band_dict.keys(), np.sum(self.availability_matrix, axis=0)
        ):
            print(f'{band_name.ljust(max_band_name_length)}: \t {count}')

        print('\nNumber of tiles available in different bands:\n')
        for bands_available, count in sorted(self.counts.items(), reverse=True):
            print(f'In {bands_available} bands: {count}')

        print(f'\nNumber of unique tiles available:\n{len(self.unique_tiles)}')


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
                logging.error(
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
    logging.info(f'Reading UNIONS catalog for tile {tile_nums}')
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

        logging.info(f'Read {len(df)} objects from UNIONS catalog for tile {tile_nums}')
    except PermissionError:
        logging.error(f'Permission error reading UNIONS catalog for tile {tile_nums}')
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
    sep_constraint = d2d < max_sep * u.arcsec
    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matching_idx = idx[sep_constraint]
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
    logging.info('Reading redshift catalog.')
    filter_coords = [
        ('ra', '>=', ra_range[0]),
        ('ra', '<=', ra_range[1]),
        ('dec', '>=', dec_range[0]),
        ('dec', '<=', dec_range[1]),
    ]
    df = pq.read_table(parquet_path, memory_map=True, filters=filter_coords).to_pandas()
    if columns:
        df = df[columns]
    logging.info(f'Read {len(df)} objects from catalog {os.path.basename(parquet_path)}.')
    return df


def add_labels(det_df, dwarfs_df, z_class_cat, lens_cat):
    """
    Add labels to detections dataframe.

    Args:
        det_df (dataframe): detections dataframe
        dwarfs_df (dataframe): known dwarfs located in the tile
        z_class_cat (dataframe): catalog dataframe with redshifts and classes
        lens_cat (dataframe): catalog dataframe with known lenses

    Returns:
        det_df (dataframe): detections dataframe with labels
    """
    logging.info('Adding labels to the detections dataframe.')
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
    det_df['class'] = np.nan
    det_df['zspec'] = np.nan
    det_df.loc[det_idx_z_class, 'class'] = label_matches_z_class['cspec'].values
    det_df.loc[det_idx_z_class, 'zspec'] = label_matches_z_class['zspec'].values

    logging.info(
        f'Number of detection matches for z/class: {len(det_idx_z_class)}, number of label matches for z/class: {len(label_matches_z_class)}'
    )

    logging.info(
        f'Found {np.count_nonzero(~np.isnan(label_matches_z_class['zspec']))} matching objects in the redshift/class catalog.'
    )

    logging.info(
        f'Added {np.count_nonzero(~np.isnan(det_df['zspec']))} to the detection dataframe.'
    )

    # read the lens catalog
    lens_df = read_parquet(lens_cat, ra_range=ra_range, dec_range=dec_range)
    # match detections to lens catalog
    det_idx_lens, label_matches_lens, _, _ = match_cats(det_df, lens_df, max_sep=1.0)
    det_idx_lens = det_idx_lens.astype(np.int32)  # make sure index is int
    # add lens labels to detections dataframe
    det_df['lens'] = np.nan
    det_df.loc[det_idx_lens, 'lens'] = 1

    logging.info(
        f'Number of detection matches for lenses: {len(det_idx_lens)}, number of label matches for lenses: {len(label_matches_lens)}'
    )

    logging.info(f'Found {len(label_matches_lens)} matching objects in the lens catalog.')

    logging.info(f'Added {np.count_nonzero(~np.isnan(det_df['lens']))} to the detection dataframe.')

    # match detections to dwarf catalog
    det_idx_lsb, _, lsb_unmatches, _ = match_cats(det_df, dwarfs_df, max_sep=10.0)
    # add lsb labels to detections dataframe
    det_df['lsb'] = np.nan
    det_df.loc[det_idx_lsb, 'lsb'] = 1

    if len(lsb_unmatches) > 0:
        logging.info(f'Found undetected but known dwarfs in tile {dwarfs_df.tile[0]}.')
        lsb_unmatches['lsb'] = 1  # dwarfs are LSB
        lsb_unmatches['class'] = 2  # dwarfs are galaxies
        # augment detections dataframe with undetected but known dwarfs
        common_columns = det_df.columns.intersection(lsb_unmatches.columns)
        logging.info(f'Common columns are: {str(common_columns)})')
        det_df = pd.concat([det_df, lsb_unmatches[common_columns]], ignore_index=True)
    logging.info('Added labels to the detections dataframe.')
    return det_df


def save_tile_cat(table_dir, tile_nums, obj_in_tile):
    """
    Save the tile catalog to a temporary file.

    Args:
        table_dir (str): path to directory where catalogs are stored
        tile_nums (tuple): tile numbers
        obj_in_tile (dataframe): objects that were cut out in the current tile
    """
    logging.info(f'Saving tile catalog for tile {tile_nums} to a temporary file.')
    temp_path = os.path.join(table_dir, f'cat_temp_{tile_nums[0]}_{tile_nums[1]}.parquet')
    obj_in_tile.to_parquet(temp_path, index=False)


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
            logging.info('Duplicate rows found. Dropping them.')

            # Remove duplicate rows
            master_table_updated = master_table_updated.drop_duplicates()
        else:
            logging.info('No duplicate rows found. Moving on.')

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
        with open(file_path, 'r') as file:
            return {line.strip() for line in file.readlines()}
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
        with open(file_path, 'a') as file:
            file.write(file_name + '\n')
    except IOError as e:
        logging.error(f'Error writing to file: {e}')


def update_h5_labels(h5_path, unions_det_dir, z_class_cat, lens_cat):
    """
    Update the labels in the cutout files in the HDF5 file based on better matching.

    Args:
        h5_path (str): path to HDF5 file
        unions_det_dir (str): path to directory containing unions catalogs
        z_class_cat (dataframe): dataframe containing the redshift and classification
        lens_cat (dataframe): dataframe containing known lenses
    """
    logging.info(f'Updating labels in {h5_path}.')
    with h5py.File(h5_path, 'r+') as hf:
        try:
            ra = hf['ra']
            dec = hf['dec']

            # create pandas dataframe
            df = pd.DataFrame({'ra': ra, 'dec': dec})
            obj_df = read_unions_cat(unions_det_dir, hf['tile'][:])

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
            obj_df['class'] = np.nan
            obj_df['zspec'] = np.nan
            det_idx = det_idx.astype(np.int32)  # make sure index is int
            obj_df.loc[det_idx, 'class'] = label_matches['cspec'].values
            obj_df.loc[det_idx, 'zspec'] = label_matches['zspec'].values

            # read the lens catalog
            lens_df = read_parquet(lens_cat, ra_range=ra_range, dec_range=dec_range)
            # match detections to lens catalog
            det_idx_lens, label_matches_lens, _, _ = match_cats(obj_df, lens_df, max_sep=1.0)
            det_idx_lens = det_idx_lens.astype(np.int32)  # make sure index is int
            # add lens labels to detections dataframe
            obj_df['lens'] = np.nan
            obj_df.loc[det_idx_lens, 'lens'] = 1

            logging.info(f'There are {len(df)} objects in the HDF5 file.')
            # change to float32 for matching
            obj_df['ra'] = obj_df['ra'].astype('float32')
            obj_df['dec'] = obj_df['dec'].astype('float32')
            obj_df['zspec'] = obj_df['zspec'].astype('float32')
            obj_df['class'] = obj_df['class'].astype('float32')
            # get part of the obj_df where ra and dec are equal to ra/dec in df
            merged_df = pd.merge(df, obj_df, on=['ra', 'dec'], how='inner')

            dataset_names = ['zspec', 'class', 'lens']  # Specify the dataset names
            # create empty dataframe
            h5_df = pd.DataFrame()
            # Check if datasets exist and create a DataFrame
            for dataset_name in dataset_names:
                try:
                    h5_df[dataset_name] = hf[dataset_name]
                except KeyError:
                    logging.warning(f"Dataset '{dataset_name}' does not exist in the HDF5 file.")
                    logging.info(f'Creating dataset {dataset_name} in the HDF5 file.')
                    h5_df[dataset_name] = np.nan * np.ones(len(df))

            z_matching = np.count_nonzero(
                merged_df['zspec'].fillna('nan') == h5_df['zspec'].fillna('nan')
            )
            class_matching = np.count_nonzero(
                merged_df['class'].fillna('nan') == h5_df['class'].fillna('nan')
            )

            lens_matching = np.count_nonzero(
                merged_df['lens'].fillna('nan') == h5_df['lens'].fillna('nan')
            )

            logging.info(
                f'Found {np.count_nonzero(~merged_df['zspec'].isna())} objects with available redshift labels.'
            )
            logging.info(
                f'Found {np.count_nonzero(~merged_df['class'].isna())} objects with available class labels.'
            )
            logging.info(
                f'Found {np.count_nonzero(~merged_df['lens'].isna())} objects that are known lenses.'
            )

            logging.info(f'Changed redshifts: {len(df)-z_matching}/{len(df)}')
            logging.info(f'Changed classes: {len(df)-class_matching}/{len(df)}')
            logging.info(f'Changed lens labels: {len(df)-lens_matching}/ {len(df)}')

            for dataset_name in dataset_names:
                if dataset_name in hf.keys():
                    hf[dataset_name][:] = merged_df[dataset_name].values
                else:
                    hf.create_dataset(
                        dataset_name, data=merged_df[dataset_name].values.astype(np.float32)
                    )

            logging.info(f'Updated labels in {h5_path}.')

        except KeyError:
            logging.error(f'No ra and dec in {h5_path}.')

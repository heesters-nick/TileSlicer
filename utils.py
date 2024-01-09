import numpy as np
import h5py
from vos import Client
import logging
import time

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
    parts = name.split(".")
    if name.startswith("calexp"):
        parts = parts[0].split("_")
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
    logging.info("Updating available tile lists from the VOSpace.")
    logging.info("Retrieving u-band tiles...")
    start_u = time.time()
    cfis_u_tiles = client.glob1("vos:cfis/tiles_DR5/", "*u.fits")
    end_u = time.time()
    logging.info(
        f"Retrieving u-band tiles completed. Took {np.round((end_u-start_u)/60, 3)} minutes."
    )
    logging.info("Retrieving g-band tiles...")
    whigs_g_tiles = client.glob1("vos:cfis/whigs/stack_images_CFIS_scheme/", "*.fits")
    end_g = time.time()
    logging.info(
        f"Retrieving g-band tiles completed. Took {np.round((end_g-end_u)/60, 3)} minutes."
    )
    logging.info("Retrieving r-band tiles...")
    cfis_lsb_r_tiles = client.glob1("vos:cfis/tiles_LSB_DR5/", "*.fits")
    end_r = time.time()
    logging.info(
        f"Retrieving r-band tiles completed. Took {np.round((end_r-end_g)/60, 3)} minutes."
    )
    logging.info("Retrieving i-band tiles...")
    ps_i_tiles = client.glob1("vos:cfis/panstarrs/DR3/tiles/", "*i.fits")
    end_i = time.time()
    logging.info(
        f"Retrieving i-band tiles completed. Took {np.round((end_i-end_r)/60, 3)} minutes."
    )
    logging.info("Retrieving z-band tiles...")
    wishes_z_tiles = client.glob1("vos:cfis/wishes_1/coadd/", "*.fits")
    end_z = time.time()
    logging.info(
        f"Retrieving z-band tiles completed. Took {np.round((end_z-end_i)/60, 3)} minutes."
    )
    if save:
        np.savetxt(path + "cfis_u_tiles.txt", cfis_u_tiles, fmt="%s")
        np.savetxt(path + "whigs_g_tiles.txt", whigs_g_tiles, fmt="%s")
        np.savetxt(path + "cfis_lsb_r_tiles.txt", cfis_lsb_r_tiles, fmt="%s")
        np.savetxt(path + "ps_i_tiles.txt", ps_i_tiles, fmt="%s")
        np.savetxt(path + "wishes_z_tiles.txt", wishes_z_tiles, fmt="%s")


def load_available_tiles(path):
    """
    Load tile lists from disk.
    :param path: path to files
    :return: lists of available tiles for the five bands
    """
    u_tiles = np.loadtxt(path + "cfis_u_tiles.txt", dtype=str)
    g_tiles = np.loadtxt(path + "whigs_g_tiles.txt", dtype=str)
    lsb_r_tiles = np.loadtxt(path + "cfis_lsb_r_tiles.txt", dtype=str)
    i_tiles = np.loadtxt(path + "ps_i_tiles.txt", dtype=str)
    z_tiles = np.loadtxt(path + "wishes_z_tiles.txt", dtype=str)

    return u_tiles, g_tiles, lsb_r_tiles, i_tiles, z_tiles


def get_tile_numbers(name):
    """
    Extract tile numbers from tile name
    :param name: .fits file name of a given tile
    :return two three digit tile numbers
    """
    parts = name.split(".")
    if name.startswith("calexp"):
        parts = parts[0].split("_")
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
        self.tile_num_sets = [
            set(map(tuple, tile_array)) for tile_array in self.all_tiles
        ]
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

    def get_availability(self, tile_number):
        try:
            index = self.unique_tiles.index(tuple(tile_number))
        except ValueError:
            # print(f'Tile number {tile_number} not available in any band.')
            return [], []
        bands_available = np.where(self.availability_matrix[index] == 1)[0]
        return [
            self.band_dict[list(self.band_dict.keys())[i]]["band"]
            for i in bands_available
        ], bands_available

    def band_tiles(self, band):
        return np.array(self.unique_tiles)[
            self.availability_matrix[:, list(self.band_dict.keys()).index(band)] == 1
        ]

    def stats(self):
        print("\nNumber of currently available tiles per band:\n")
        max_band_name_length = max(map(len, self.band_dict.keys()))  # for output format
        for band_name, count in zip(
            self.band_dict.keys(), np.sum(self.availability_matrix, axis=0)
        ):
            print(f"{band_name.ljust(max_band_name_length)}: \t {count}")

        print("\nNumber of tiles available in different bands:\n")
        for bands_available, count in sorted(self.counts.items(), reverse=True):
            print(f"In {bands_available} bands: {count}")

        print(f"\nNumber of unique tiles available:\n{len(self.unique_tiles)}")


def read_h5(cutout_dir):
    """
    Reads cutout data from HDF5 file
    :param cutout_dir: cutout directory
    :return: cutout data
    """
    with h5py.File(cutout_dir, "r") as f:
        # Create empty dictionaries to store data for each group
        cutout_data = {}

        # Loop through datasets
        for dataset_name in f:
            data = np.array(f[dataset_name])
            cutout_data[dataset_name] = data
    return cutout_data

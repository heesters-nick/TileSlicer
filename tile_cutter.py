import os
import numpy as np
import time
import pandas as pd
import argparse
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import Cutout2D
from utils import update_available_tiles, extract_tile_numbers, load_available_tiles, TileAvailability, read_h5
from kd_tree import query_tree, TileWCS, relate_coord_tile, build_tree
from plotting import plot_cutout
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import h5py
from datetime import timedelta
import random
from vos import Client
client = Client()

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .

band_dict = {'cfis-u': {'name': 'CFIS', 'band': 'u', 'vos': 'vos:cfis/tiles_DR5/', 'suffix': '.u.fits', 'delimiter': '.', 'fits_ext': 0},
             'whigs-g': {'name': 'calexp-CFIS', 'band': 'g', 'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme/', 'suffix': '.fits', 'delimiter': '_', 'fits_ext': 1},
             'cfis_lsb-r': {'name': 'CFIS_LSB', 'band': 'r', 'vos': 'vos:cfis/tiles_LSB_DR5/', 'suffix': '.r.fits', 'delimiter': '.', 'fits_ext': 0},
             'ps-i': {'name': 'PS-DR3', 'band': 'i', 'vos': 'vos:cfis/panstarrs/DR3/tiles/', 'suffix': '.i.fits', 'delimiter': '.', 'fits_ext': 0},
             'wishes-z': {'name': 'WISHES', 'band': 'z', 'vos': 'vos:cfis/wishes_1/coadd/', 'suffix': '.z.fits', 'delimiter': '.', 'fits_ext': 1}}

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
# define the minimum number of bands that should be available for a tile
band_constraint = 3
# download the tiles
download_tiles = True
# Plot cutouts from one of the tiles after execution
with_plot = True
# Show plot
show_plot = False
# Save plot
save_plot = True

# paths
# define the root directory
parent_directory = '/home/nick/astro/TileSlicer/'
cat_directory = os.path.join(parent_directory, 'tables/')
os.makedirs(cat_directory, exist_ok=True)
catalog_script = pd.read_csv(cat_directory+'NGC5485_dwarfs.csv')
ra_key_script, dec_key_script, id_key_script = 'ra', 'dec', 'ID'
# define where the information about the currently available tiles should be saved
tile_info_directory = os.path.join(parent_directory, 'tile_info/')
os.makedirs(tile_info_directory, exist_ok=True)
# define where the tiles should be saved
download_directory = os.path.join(parent_directory, 'data/')
os.makedirs(download_directory, exist_ok=True)
# define where the cutouts should be saved
cutout_directory = os.path.join(parent_directory, 'h5_files/')
os.makedirs(cutout_directory, exist_ok=True)
# define where figures should be saved
figure_directory = os.path.join(parent_directory, 'figures/')
os.makedirs(figure_directory, exist_ok=True)

# tile parameters
cutout_size = 200
h5_filename = 'cutout_stacks_ugriz_lsb_200x200'
num_workers = 9  # specifiy the number of parallel workers following machine capabilities


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
    for i, obj_coord in enumerate(coord_c):
        tile_numbers, _ = query_tree(available_tiles, np.array([obj_coord.ra.deg, obj_coord.dec.deg]), tile_info_dir)
        tiles_matching_catalog[i] = tile_numbers
        wcs = TileWCS()
        wcs.set_coords(relate_coord_tile(nums=tile_numbers))
        pix_coord = skycoord_to_pixel(obj_coord, wcs.wcs_tile, origin=1)
        pix_coords[i] = pix_coord

    # add tile numbers and pixel coordinates to catalog
    catalog['tile'] = tiles_matching_catalog
    catalog['x'] = pix_coords[:, 0]
    catalog['y'] = pix_coords[:, 1]
    unique_tiles = list(set(tiles_matching_catalog))
    tiles_x_bands = [tile for tile in unique_tiles if len(availability.get_availability(tile)[1]) >= band_constr]

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
    for band in np.array(list(band_dict.keys()))[avail_idx]:
        vos_dir = in_dict[band]['vos']
        prefix = in_dict[band]['name']
        suffix = in_dict[band]['suffix']
        delimiter = in_dict[band]['delimiter']
        tile_dir = download_dir + f'{tile_numbers[0]}_{tile_numbers[1]}'
        os.makedirs(tile_dir, exist_ok=True)
        tile_fitsfilename = f'{prefix}{delimiter}{tile_numbers[0]}{delimiter}{tile_numbers[1]}{suffix}'
        # use a temporary name while the file is downloading
        temp_name = '.'.join(tile_fitsfilename.split('.')[:-1]) + '_temp.fits'

        # Check if the directory exists, and create it if not
        if os.path.exists(os.path.join(tile_dir, tile_fitsfilename)):
            print(f'File {tile_fitsfilename} was already downloaded.')
        else:
            print(f'Downloading {tile_fitsfilename}..')
            try:
                if method == 'command':
                    # command line
                    os.system(f'vcp -v {vos_dir + tile_fitsfilename} {os.path.join(tile_dir, temp_name)}')
                else:
                    # API
                    client.copy(os.path.join(vos_dir, tile_fitsfilename), os.path.join(tile_dir, temp_name))
                os.rename(os.path.join(tile_dir, temp_name), os.path.join(tile_dir, tile_fitsfilename))
            except Exception as e:
                print(e)
                return False
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
    img_cutout = Cutout2D(data, (x, y), size, mode="partial", fill_value=0).data

    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05 * size ** 2 or np.count_nonzero(img_cutout) == 0:
        return np.zeros((size, size))  # Don't use this cutout

    img_cutout[np.isnan(img_cutout)] = 0

    return img_cutout


def make_cutouts_all_bands(availability, tile, obj_in_tile, download_dir, in_dict, size):
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
    avail_idx = availability.get_availability(tile)[1]
    cutout = np.zeros((len(obj_in_tile), len(in_dict), size, size))
    tile_dir = download_dir+f'{tile[0]}_{tile[1]}'
    for j, band in enumerate(np.array(list(in_dict.keys()))[avail_idx]):
        prefix = in_dict[band]['name']
        suffix = in_dict[band]['suffix']
        delimiter = in_dict[band]['delimiter']
        fits_ext = in_dict[band]['fits_ext']
        tile_fitsfilename = f'{prefix}{delimiter}{tile[0]}{delimiter}{tile[1]}{suffix}'
        with fits.open(os.path.join(tile_dir, tile_fitsfilename), memmap=True) as hdul:
            data = hdul[fits_ext].data
        for i, (x, y) in enumerate(zip(obj_in_tile.x.values, obj_in_tile.y.values)):
            cutout[i, j] = make_cutout(data, x, y, size)
    return cutout


def save_to_h5(stacked_cutout, tile_numbers, ids, ras, decs, save_path):
    """
    Save cutout data including metadata to file.
    :param stacked_cutout: stacked numpy array of the image data in different bands
    :param tile_numbers: tile numbers
    :param ids: object IDs
    :param ras: right ascension coordinate array
    :param decs: declination coordinate array
    :param save_path: save path
    :return: pass
    """
    print('Saving file: {}'.format(save_path))
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(save_path, 'w', libver='latest') as hf:
        hf.create_dataset('images', data=stacked_cutout.astype(np.float32))
        hf.create_dataset('tile', data=np.asarray(tile_numbers), dtype=np.int32)
        hf.create_dataset('cfis_id', data=np.asarray(ids, dtype='S'), dtype=dt)
        hf.create_dataset('ra', data=ras.astype(np.float32))
        hf.create_dataset('dec', data=decs.astype(np.float32))
    pass


def process_tile(availability, tile, catalog, id_key, ra_key, dec_key, cutout_dir, h5_name, download_dir, in_dict, size):
    """
    Process a tile, create cutouts in all bands, save cutouts and metadata to hdf5 file
    :param availability: object to retrieve available tiles
    :param tile: tile numbers
    :param catalog: object catalog
    :param id_key: id key in the catalog
    :param ra_key: ra key in the catalog
    :param dec_key: dec key in the catalog
    :param cutout_dir: cutout directory
    :param h5_name: hdf5 file name
    :param download_dir: tile directory
    :param in_dict: band dictionary
    :param size: cutout size
    :return image cutout in available bands, array with shape: (n_bands, cutout_size, cutout_size)
    """
    save_path = os.path.join(cutout_dir, h5_name + f'_{tile[0]}_{tile[1]}.h5')
    obj_in_tile = catalog.loc[catalog['tile'] == tile]
    cutout = make_cutouts_all_bands(availability, tile, obj_in_tile, download_dir, in_dict, size)
    save_to_h5(cutout, tile, obj_in_tile[id_key].values, obj_in_tile[ra_key].values, obj_in_tile[dec_key].values, save_path)
    return cutout


def main(cat_default, ra_key_default, dec_key_default, id_key_default, tile_info_dir, in_dict, at_least_key, band_constr, download_dir, cutout_dir, figure_dir, size, h5_name, workers, update, show_stats, dl_tiles, build_kdtree, coordinates=None, dataframe_path=None, ra_key=None, dec_key=None, id_key=None, show_plt=False, save_plt=False):
    if coordinates is not None:
        coordinates = coordinates[0]
        if (len(coordinates) == 0) or len(coordinates) % 2 != 0:
            raise ValueError('Provide even number of coordinates.')
        ras, decs, ids = coordinates[::2], coordinates[1::2], list(np.arange(1, len(coordinates)//2 + 1))
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        df_coordinates = pd.DataFrame({id_key: ids, ra_key: ras, dec_key: decs})

        formatted_coordinates = " ".join([f"({ra}, {dec})" for ra, dec in zip(ras, decs)])
        print(f'Coordinates received from the command line: {formatted_coordinates}')

        catalog = df_coordinates
        df_coordinates.to_csv('df_coordinates_test.csv', index=False)
        coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')
    elif dataframe_path is not None:
        print(f'Dataframe received from command line.')
        catalog = pd.read_csv(dataframe_path)
        coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')
    else:
        print('No coordinates or DataFrame provided. Using coordinates from default DataFrame.')
        catalog = cat_default
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')
    if update:
        update_available_tiles(tile_info_dir)

    u, g, lsb_r, i, z = extract_tile_numbers(load_available_tiles(tile_info_dir))
    all_bands = [u, g, lsb_r, i, z]
    availability = TileAvailability(all_bands, in_dict, at_least_key)
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    if show_stats:
        availability.stats()

    unique_tiles, tiles_x_bands = tile_finder(availability, catalog, coord_c, tile_info_dir, band_constr)

    for tile in unique_tiles:
        bands = availability.get_availability(tile)[0]
        print(f'Tile {tile} is available in {len(bands)} bands: {bands}')

    if dl_tiles:
        print('Downloading the tiles in the available bands..')
        start_download = time.time()
        for tile in tiles_x_bands:
            if download_tile_for_bands(availability, tile, in_dict, download_dir, method='command'):
                print(f'Tile downloaded in all available bands. Took {(time.time() - start_download) / 60} minutes.')

    # log tile processing
    successful_tiles_count = 0
    total_cutouts_count = 0
    failed_tiles = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_tile = {
            executor.submit(process_tile, availability, tile, catalog, id_key, ra_key, dec_key, cutout_dir, h5_name, download_dir,
                            in_dict, size): tile for tile in tiles_x_bands}

        for future in concurrent.futures.as_completed(future_to_tile):
            tile = future_to_tile[future]
            try:
                cutout = future.result()
                if cutout is not None:
                    total_cutouts_count += cutout.shape[0]
                    successful_tiles_count += 1
            except Exception as e:
                print(f"Failed to process tile {tile}: {str(e)}")
                failed_tiles.append(tile)

    print(f'\nProcessing report:\nTiles processed: {len(tiles_x_bands)}\nCutouts created: {total_cutouts_count}'
          f'\nTiles failed: {len(failed_tiles)}/{len(tiles_x_bands)}')
    if len(failed_tiles) != 0:
        print(f'Processing error in tiles: {failed_tiles}.')

    if with_plot:
        random_tile_index = random.randint(0, len(tiles_x_bands))
        for idx in range(len(tiles_x_bands)):
            cutout_path = os.path.join(cutout_dir, h5_name +
                                       f'_{tiles_x_bands[idx][0]}_{tiles_x_bands[idx][1]}.h5')
            cutout = read_h5(cutout_path)

            plot_cutout(cutout, in_dict, figure_dir, show_plot=show_plt, save_plot=save_plt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', nargs='+', type=float, action='append', metavar=('ra', 'dec'),
                        help='list of pairs of coordinates to make cutouts from')
    parser.add_argument('--dataframe', type=str,
                        help='path to a CSV file containing the DataFrame')
    parser.add_argument('--ra_key', type=str,
                        help='right ascension key in the DataFrame')
    parser.add_argument('--dec_key', type=str,
                        help='declination key in the DataFrame')
    parser.add_argument('--id_key', type=str,
                        help='id key in the DataFrame')
    args = parser.parse_args()

    arg_dict_main = {
        'cat_default': catalog_script,
        'ra_key_default': ra_key_script,
        'dec_key_default': dec_key_script,
        'id_key_default': id_key_script,
        'tile_info_dir': tile_info_directory,
        'in_dict': band_dict,
        'at_least_key': at_least,
        'band_constr': band_constraint,
        'download_dir': download_directory,
        'cutout_dir': cutout_directory,
        'figure_dir': figure_directory,
        'size': cutout_size,
        'h5_name': h5_filename,
        'workers': num_workers,
        'update': update_tiles,
        'show_stats': show_tile_statistics,
        'dl_tiles': download_tiles,
        'build_kdtree': build_new_kdtree,
        'coordinates': args.coordinates,
        'dataframe_path': args.dataframe,
        'ra_key': args.ra_key,
        'dec_key': args.dec_key,
        'id_key': args.id_key,
        'show_plt': show_plot,
        'save_plt': save_plot
    }

    start = time.time()
    main(**arg_dict_main)
    end = time.time()
    elapsed = end-start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = elapsed_string.split(':')[0], elapsed_string.split(':')[1], elapsed_string.split(':')[2]
    print(f'Done! Execution took {hours} hours, {minutes} minutes, and {seconds} seconds.')

import numpy as np
from scipy.spatial import cKDTree
import joblib
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from utils import relate_coord_tile


def build_tree(tiles, tile_info_dir, save=True):
    """
    Build a kd tree from the input tiles to efficiently query object positions.
    :param tiles: list of unique tile number pairs
    :param tile_info_dir: path to save the tree
    :param save: save the built tree to file
    :return: pass
    """
    tile_coords = np.array([relate_coord_tile(nums=num) for num in tiles])
    tile_coords_c = SkyCoord(tile_coords[:, 0], tile_coords[:, 1], unit='deg', frame='icrs')
    tile_coords_xyz = np.array([x.cartesian.xyz.value for x in tile_coords_c])
    tree = cKDTree(tile_coords_xyz)
    if save:
        joblib.dump(tree, tile_info_dir + 'kdtree_xyz.joblib')
    pass


def query_tree(tiles, coords, tile_info_dir):
    """
    Query the kd tree to find what tile an object is in.
    :param tiles: list of tile numbers as tuples
    :param coords: ra, dec of object to query
    :param tile_info_dir: path to save and load the tree
    :return tile name and distance object - nearest tile center
    """
    loaded_tree = joblib.load(tile_info_dir + 'kdtree_xyz.joblib')
    try:
        tile_name, dist = find_tile(loaded_tree, tiles, coords)
        return tile_name, dist
    except ValueError as e:
        return np.nan, f'Error: {e}'


class TileWCS:
    def __init__(self, wcs_keywords={}):
        wcs_keywords.update({
            'NAXIS': 2,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRVAL1': 0,
            'CRVAL2': 0,
            'CRPIX1': 5000.0,
            'CRPIX2': 5000.0,
            'CD1_1': -5.160234650248e-05,
            'CD1_2': 0.0,
            'CD2_1': 0.0,
            'CD2_2': 5.160234650248e-05,
            'NAXIS1': 10000,
            'NAXIS2': 10000
        })

        self.wcs_tile = WCS(wcs_keywords)

    def set_coords(self, coords):
        self.wcs_tile.wcs.crval = [coords[0], coords[1]]


def find_tile(tree, tiles, object_coord):
    """
    Query the tree and find the tile the object is in.
    :param tree: kd tree
    :param tiles: list of unique tiles in the survey
    :param object_coord: coordinates of the object we want to find a tile for
    :return: tile numbers of the matching tile
    """
    coord_c = SkyCoord(object_coord[0], object_coord[1], unit='deg', frame='icrs')
    coord_xyz = coord_c.cartesian.xyz.value
    dists, indices = tree.query(coord_xyz, k=4)
    wcs = TileWCS()
    for dist, idx in zip(dists, indices):
        wcs.set_coords(relate_coord_tile(nums=tiles[idx]))
        if wcs.wcs_tile.footprint_contains(coord_c):
            return tiles[idx], dist
    raise ValueError('Object could not be assigned to a tile.')

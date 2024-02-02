import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm


def plot_cutout(
    cutout, in_dict, figure_dir, random_obj_index=None, show_plot=True, save_plot=False
):
    """
    Plot cutouts in available bands.
    :param cutout: cutout data
    :param in_dict: band dictionary
    :param figure_dir: figure path
    :param random_obj_index: random object index
    :param show_plot: show plot
    :param save_plot: save plot
    :return: cutout plot
    """
    title_map = {
        'cfis-u': 'CFHT-u',
        'whigs-g': 'HSC-g',
        'cfis_lsb-r': 'CFHT-LSB-r',
        'ps-i': 'PS-i',
        'wishes-z': 'HSC-z',
    }
    shape = cutout['images'].shape
    if random_obj_index:
        image_data = cutout['images'][random_obj_index].reshape(1, shape[1], shape[2], shape[3])
        obj_ids = [cutout['cfis_id'][random_obj_index].decode('utf-8')]
    else:
        image_data = cutout['images']
        obj_ids = np.array([x.decode('utf-8') for x in cutout['cfis_id']])
    tile = cutout['tile']
    n_objects, n_bands = image_data.shape[0], image_data.shape[1]
    fig, axes = plt.subplots(n_objects, n_bands, figsize=(n_bands * 4, n_objects * 4))

    # Make sure axes is always a 2D array
    if n_objects == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop through objects and filter bands, and plot each image
    for i in range(n_objects):  # Number of objects
        for j, band in enumerate(in_dict.keys()):  # Number of filter bands
            ax = axes[i, j]
            # Get the image data for the current object and filter band
            image = image_data[i, j]
            # Display the image
            image[image < -10] = np.nan
            image[image > 1300] = np.nanmax(image)
            norm = simple_norm(image, 'sqrt', percent=98.0)
            image[np.isnan(image)] = 0.0
            ax.imshow(image, norm=norm, cmap='viridis', origin='lower')
            if i == 0:
                ax.set_title(f'{title_map[band]}', fontsize=45, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{obj_ids[i]}', fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if save_plot:
        plt.savefig(
            os.path.join(figure_dir, f'cutouts_tile_{tile[0]}_{tile[1]}.pdf'),
            bbox_inches='tight',
            dpi=300,
        )
    if show_plot:
        plt.show()
    else:
        plt.close()

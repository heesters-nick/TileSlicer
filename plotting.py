import numpy as np
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import os


def plot_cutout(cutout, in_dict, figure_dir, show_plot=True, save_plot=False):
    """
    Plot cutouts in available bands.
    :param cutout: cutout data
    :param in_dict: band dictionary
    :param figure_dir: figure path
    :param show_plot: show plot
    :param save_plot: save plot
    :return: cutout plot
    """
    image_data = cutout['images']
    tile = cutout['tile']
    obj_ids = np.array([x.decode('utf-8') for x in cutout['cfis_id']])
    n_objects, n_bands = image_data.shape[0], image_data.shape[1]
    fig, axes = plt.subplots(n_objects, n_bands, figsize=(n_bands*4, n_objects*4))

    # Make sure axes is always a 2D array
    if n_objects == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop through objects and filter bands, and plot each image
    for i in range(n_objects):  # Number of objects
        for j, band in enumerate(in_dict.keys()):  # Number of filter bands
            filter_name = in_dict[band]['band']
            ax = axes[i, j]

            # Get the image data for the current object and filter band
            image = image_data[i, j]
            # Display the image
            norm = simple_norm(image, 'sqrt', percent=98.)
            ax.imshow(image, norm=norm, cmap='gray_r', origin='lower')  # Adjust the cmap as needed
            ax.set_title(f'{obj_ids[i]}, {filter_name}')

            # Optionally, you can turn off axis labels if they are not needed
            ax.axis('off')

    plt.tight_layout()

    if save_plot:
        plt.savefig(os.path.join(figure_dir, f'cutouts_tile_{tile[0]}_{tile[1]}.pdf'), bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, binary_fill_holes, label

from data_utils import adjust_flux_with_zp, find_band_indices


def plot_cutout(
    cutout,
    in_dict,
    figure_dir,
    rgb_bands=['i', 'r', 'g'],
    random_obj_index=None,
    show_plot=True,
    save_plot=False,
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

    def local_warn_handler(message, category, filename, lineno, file=None, line=None):
        # Custom message with context about the image
        log = (
            f'Warning in tile {cutout["tile"]}: {filename}:{lineno}: {category.__name__}: {message}'
        )
        logging.warning(log)  # Log the warning with contextual info

    title_map = {
        'cfis-u': 'CFHT-u',
        'whigs-g': 'HSC-g',
        'cfis_lsb-r': 'CFHT-LSB-r',
        'ps-i': 'PS-i',
        'wishes-z': 'HSC-z',
        'ps-z': 'PS-z',
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
    fig, axes = plt.subplots(
        n_objects, n_bands + 1, figsize=((n_bands + 1) * 4, n_objects * 4)
    )  # +1 for RGB

    # Make sure axes is always a 2D array
    if n_objects == 1:
        axes = np.expand_dims(axes, axis=0)

    # Loop through objects and filter bands, and plot each image
    for i in range(n_objects):  # Number of objects
        for j in range(n_bands + 1):  # Number of filter bands + 1 for RGB
            ax = axes[i, j]  # type: ignore
            if j < n_bands:
                band = list(in_dict.keys())[j]
                # Get the image data for the current object and filter band
                image = image_data[i, j]
                # Display the image
                epsilon = 1e-10
                image[image == 0.0] = epsilon
                image[image == 0] = epsilon
                image[image < -10] = epsilon
                image[image > 1300] = np.nanmax(image)
                image = np.nan_to_num(image, nan=epsilon, posinf=epsilon, neginf=epsilon)
                try:
                    norm = simple_norm(image, 'sqrt', percent=98.0)
                finally:
                    warnings.showwarning = local_warn_handler
                # image[np.isnan(image)] = 0.0
                ax.imshow(image, norm=norm, cmap='viridis', origin='lower')
                if i == 0:
                    ax.set_title(f'{title_map[band]}', fontsize=45, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(f'{obj_ids[i]}', fontsize=20)
            else:
                rgb = cutout_rgb(
                    cutout,
                    i,
                    rgb_bands,
                    in_dict,
                    figure_dir,
                    plot_rgb_cutout=False,
                    save_rgb_cutout=False,
                )
                ax.imshow(rgb)
                if i == 0:
                    ax.set_title(f'RGB ({"".join(rgb_bands)})', fontsize=45, fontweight='bold')
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


def rgb_image(
    img,
    scaling_type='linear',  # Default to 'asinh', can also use 'linear'
    stretch=0.5,
    Q=10.0,
    m=0.0,
    ceil_percentile=99.8,
    dtype=np.uint8,
    do_norm=True,
    gamma=0.35,
    scale_red=1.0,
    scale_green=1.0,
    scale_blue=1.0,
):
    """
    Create an RGB image from three bands of data. The bands are assumed to be in the order RGB.

    Args:
        img (numpy.ndarray): image data in three bands, (size,size,3)
        scaling_type (str, optional): scaling type, use asinh or linear. Defaults to 'linear'.
        Q (float, optional): asinh softening parameter. Defaults to 10.
        m (float, optional): intensity that should be mapped to black. Defaults to 0.
        ceil_percentile (float, optional): percentile used to normalize the data. Defaults to 99.8.
        dtype (type, optional): dtype. Defaults to np.uint8.
        do_norm (bool, optional): normalize the data. Defaults to True.
        gamma (float, optional): perform gamma correction. Defaults to 0.35.
        scale_red (float, optional): scale contribution of red band. Defaults to 1.0.
        scale_green (float, optional): scale contribution of green band. Defaults to 1.0.
        scale_blue (float, optional): scale contribution of blue band. Defaults to 1.0.
    """

    def norm(red, green, blue, scale_red, scale_green, scale_blue):
        red = red / np.nanpercentile(red, ceil_percentile)
        green = green / np.nanpercentile(green, ceil_percentile)
        blue = blue / np.nanpercentile(blue, ceil_percentile)

        max_dtype = np.iinfo(dtype).max
        red = np.clip((red * max_dtype), 0, max_dtype)
        green = np.clip((green * max_dtype), 0, max_dtype)
        blue = np.clip((blue * max_dtype), 0, max_dtype)

        image[:, :, 0] = scale_red * red  # Red
        image[:, :, 1] = scale_green * green  # Green
        image[:, :, 2] = scale_blue * blue  # Blue
        return image

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Compute average intensity before scaling choice
    i_mean = (red + green + blue) / 3.0

    length, width = green.shape
    image = np.empty([length, width, 3], dtype=dtype)

    if scaling_type == 'asinh':
        # Apply asinh scaling
        red = red * np.arcsinh(stretch * Q * (i_mean - m)) / (Q * i_mean)
        green = green * np.arcsinh(stretch * Q * (i_mean - m)) / (Q * i_mean)
        blue = blue * np.arcsinh(stretch * Q * (i_mean - m)) / (Q * i_mean)
    elif scaling_type == 'linear':
        # Apply linear scaling
        max_val = np.nanpercentile(i_mean, ceil_percentile)
        red = (red / max_val) * np.iinfo(dtype).max
        green = (green / max_val) * np.iinfo(dtype).max
        blue = (blue / max_val) * np.iinfo(dtype).max
    else:
        raise ValueError(f'Unknown scaling type: {scaling_type}')

    # Optionally apply gamma correction
    if gamma is not None:
        red = np.clip((red**gamma), 0, np.iinfo(dtype).max)
        green = np.clip((green**gamma), 0, np.iinfo(dtype).max)
        blue = np.clip((blue**gamma), 0, np.iinfo(dtype).max)

    if do_norm:
        return norm(red, green, blue, scale_red, scale_green, scale_blue)
    else:
        return np.stack([red, green, blue], axis=-1)


def find_percentile_from_target(cutouts, target_value):
    """
    Determines the first percentile from 100 to 0 where the value is less than or equal to the target value

    Args:
        cutouts (list): list of numpy.ndarrays for each band in the order [i, r, g]
        target_value (float): target value to compare against

    Returns:
        dict: dictionary containing the first percentiles where values are <= target_value for each band
    """
    results = {}
    bands = ['R', 'G', 'B']  # Define band names according to the order of input arrays
    percentiles = np.arange(100, 0, -0.01)  # Creating percentiles from 100 to 0 with 0.01 steps

    for band, cutout in zip(bands, cutouts):
        # We calculate values at each percentile
        values_at_percentiles = np.nanpercentile(cutout, percentiles)

        # Check for the first value that is <= target value
        idx = np.where(values_at_percentiles <= target_value)[0]
        if idx.size > 0:
            results[band] = percentiles[idx[0]]
        else:
            results[band] = 100.0

    return results


def desaturate(image, saturation_percentile, interpolate_neg=False, min_size=10, fill_holes=True):
    """
    Desaturate saturated pixels in an image using interpolation.

    Args:
        image (numpy.ndarray): single band image data
        saturation_percentile (float): percentile to use as saturation threshold
        interpolate_neg (bool, optional): interpolate patches of negative values. Defaults to False.
        min_size (int, optional): number of pixels in a patch to perform interpolation of neg values. Defaults to 10.
        fill_holes (bool, optional): fill holes in generated saturation mask. Defaults to True.

    Returns:
        numpy.ndarray: desaturated image, mask of saturated pixels
    """
    # Assuming image is a 2D numpy array for one color band
    # Identify saturated pixels
    mask = image >= np.nanpercentile(image, saturation_percentile)
    mask = binary_dilation(mask, iterations=2)

    if interpolate_neg:
        neg_mask = image <= 0.9

        labeled_array, num_features = label(neg_mask)  # type: ignore
        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

        # Prepare to accumulate a total mask
        total_feature_mask = np.zeros_like(image, dtype=np.float64)

        # Loop through all labels to find significant components
        for component_label in range(1, num_features + 1):  # Start from 1 to skip background
            if component_sizes[component_label] >= min_size:
                # Create a binary mask for this component
                component_mask = labeled_array == component_label
                # add component mask to component masks
                # Accumulate the upscaled feature mask
                total_feature_mask |= component_mask

        total_feature_mask = binary_dilation(total_feature_mask, iterations=1)
        mask = np.logical_or(mask, total_feature_mask)

    if fill_holes:
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
        filled_padded_mask = binary_fill_holes(padded_mask)
        if filled_padded_mask is not None:
            mask = filled_padded_mask[1:-1, 1:-1]

    y, x = np.indices(image.shape)

    # Coordinates of non-saturated pixels
    x_nonsat = x[np.logical_not(mask)]
    y_nonsat = y[np.logical_not(mask)]
    values_nonsat = image[np.logical_not(mask)]

    # Coordinates of saturated pixels
    x_sat = x[mask]
    y_sat = y[mask]

    # Interpolate to find values at the positions of the saturated pixels
    interpolated_values = griddata(
        (y_nonsat.flatten(), x_nonsat.flatten()),  # points
        values_nonsat.flatten(),  # values
        (y_sat.flatten(), x_sat.flatten()),  # points to interpolate
        method='linear',  # 'linear', 'nearest' or 'cubic'
    )
    # If any of the interpolated values are NaN, use nearest interpolation
    if np.any(np.isnan(interpolated_values)):
        interpolated_values = griddata(
            (y_nonsat.flatten(), x_nonsat.flatten()),  # points
            values_nonsat.flatten(),  # values
            (y_sat.flatten(), x_sat.flatten()),  # points to interpolate
            method='nearest',  # 'linear', 'nearest' or 'cubic'
        )

    # Replace saturated pixels in the image
    new_image = image.copy()
    new_image[y_sat, x_sat] = interpolated_values

    return new_image, mask


def cutout_rgb(
    cutout, obj_idx, bands, in_dict, save_dir, plot_rgb_cutout=False, save_rgb_cutout=False
):
    """
    Create an RGB image from the cutout data and save or plot it.

    Args:
        cutout (dict): dictionary containing cutout data
        obj_idx (int): object index in the cutout data
        bands (list): list of bands to use for the RGB image
        in_dict (dict): band dictionary
        save_dir (str): directory to save the RGB image
        plot_rgb_cutout (bool, optional): plot the cutout. Defaults to False.
        save_rgb_cutout (bool, optional): save the cutout. Defaults to False.

    Returns:
        PIL image: image cutout
    """
    band_idx = find_band_indices(in_dict, bands)
    cutout_rgb = cutout['images'][obj_idx][band_idx]

    cutout_red = cutout_rgb[2]  # R
    cutout_green = cutout_rgb[1]  # G
    cutout_blue = adjust_flux_with_zp(cutout_rgb[0], 27.0, 30.0)  # B

    percentile = 99.9
    saturation_percentile_threshold = 1000.0
    high_saturation_threshold = 20000.0
    interpolate_neg = False
    min_size = 1000
    percentile_red = np.nanpercentile(cutout_red, percentile)
    percentile_green = np.nanpercentile(cutout_green, percentile)
    percentile_blue = np.nanpercentile(cutout_blue, percentile)

    #     print(f'{percentile} percentile r: {percentile_r}')
    #     print(f'{percentile} percentile g: {percentile_g}')
    #     print(f'{percentile} percentile i: {percentile_i}')

    if np.any(
        np.array([percentile_red, percentile_green, percentile_blue])
        > saturation_percentile_threshold
    ):
        # If any band is highly saturated choose a lower percentile target to bring out more lsb features
        if np.any(
            np.array([percentile_red, percentile_green, percentile_blue])
            > high_saturation_threshold
        ):
            percentile_target = 200.0
        else:
            percentile_target = 1000.0

        # Find individual saturation percentiles for each band
        percentiles = find_percentile_from_target(
            [cutout_red, cutout_green, cutout_blue], percentile_target
        )
        cutout_red_desat, _ = desaturate(
            cutout_red,
            saturation_percentile=percentiles['R'],  # type: ignore
            interpolate_neg=interpolate_neg,
            min_size=min_size,
        )
        cutout_green_desat, _ = desaturate(
            cutout_green,
            saturation_percentile=percentiles['G'],  # type: ignore
            interpolate_neg=interpolate_neg,
            min_size=min_size,
        )
        cutout_blue_desat, _ = desaturate(
            cutout_blue,
            saturation_percentile=percentiles['B'],  # type: ignore
            interpolate_neg=interpolate_neg,
            min_size=min_size,
        )

        rgb = np.stack(
            [cutout_red_desat, cutout_green_desat, cutout_blue_desat], axis=-1
        )  # Stack data in [R, G, B] order
    else:
        rgb = np.stack([cutout_red, cutout_green, cutout_blue], axis=-1)

    # Create RGB image
    img_linear = rgb_image(
        rgb,
        scaling_type='linear',
        stretch=0.9,
        Q=5,
        ceil_percentile=99.8,
        dtype=np.uint8,
        do_norm=True,
        gamma=0.35,
        scale_red=1.0,
        scale_green=1.0,
        scale_blue=1.0,
    )

    img_linear = Image.fromarray(img_linear)
    img_linear = img_linear.transpose(Image.FLIP_TOP_BOTTOM)
    obj_id = cutout['cfis_id'][obj_idx].decode('utf-8').replace(' ', '_')

    if save_rgb_cutout:
        img_linear.save(os.path.join(save_dir, f'{obj_id}.png'))
    if plot_rgb_cutout:
        plt.figure(figsize=(8, 8))
        plt.imshow(img_linear)
        plt.title(obj_id, fontsize=20)
        plt.gca().axis('off')
        plt.show()

    return img_linear

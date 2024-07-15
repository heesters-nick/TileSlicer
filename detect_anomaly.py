import os

import matplotlib.pyplot as plt
import numpy as np
import pywt
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.visualization import simple_norm
from photutils.background import Background2D, SExtractorBackground
from scipy.ndimage import binary_dilation, label


def load_fits(file_path, ext=0):
    if ('calexp' in file_path) or ('WISHES' in file_path):
        ext = 1
    with fits.open(file_path, memmap=True) as hdul:
        data = hdul[ext].data.astype(np.float64)  # type: ignore
        header = hdul[ext].header  # type: ignore
    return data, header


def estimate_background(data, bg_mesh=128):
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = SExtractorBackground()
    bkg = Background2D(
        data,
        (bg_mesh, bg_mesh),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    return bkg.background_rms_median


def detect_anomaly(
    image,
    figure_path,
    file_out,
    zero_threshold=0.0025,
    min_size=50,
    bkg_rms=0.0,
    replace_anomaly=False,
    dilate_mask=False,
    dilation_iters=1,
    show_plot=False,
    save_plot=False,
):
    # Takes both data or path to file
    if isinstance(image, str):
        image, header = load_fits(image)

    # Perform a 2D Discrete Wavelet Transform using Haar wavelets
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs  # Decomposition into approximation and details

    # Create binary masks where wavelet coefficients are below the threshold
    mask_horizontal = np.abs(cH) <= zero_threshold
    mask_vertical = np.abs(cV) <= zero_threshold
    mask_diagonal = np.abs(cD) <= zero_threshold

    masks = [mask_diagonal, mask_horizontal, mask_vertical]

    global_mask = np.zeros_like(image, dtype=bool)
    component_masks = np.zeros((3, cA.shape[0], cA.shape[1]), dtype=bool)
    anomalies = np.zeros(3, dtype=bool)
    for i, mask in enumerate(masks):
        # Apply connected-component labeling to find connected regions in the mask
        labeled_array, num_features = label(mask)  # type: ignore

        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

        anomaly_detected = np.any(component_sizes[1:] >= min_size)
        anomalies[i] = anomaly_detected

        if not anomaly_detected:
            continue

        # Prepare to accumulate a total mask
        total_feature_mask = np.zeros_like(image, dtype=bool)

        # Loop through all labels to find significant components
        for component_label in range(1, num_features + 1):  # Start from 1 to skip background
            if component_sizes[component_label] >= min_size:
                # Create a binary mask for this component
                component_mask = labeled_array == component_label
                # add component mask to component masks
                component_masks[i] |= component_mask
                # Upscale the mask to match the original image dimensions
                upscaled_mask = np.kron(component_mask, np.ones((2, 2), dtype=bool))
                # Accumulate the upscaled feature mask
                total_feature_mask |= upscaled_mask

        # Accumulate global mask
        global_mask |= total_feature_mask
        # Dilate the masks to catch some odd pixels on the outskirts of the anomaly
        if dilate_mask:
            global_mask = binary_dilation(global_mask, iterations=dilation_iters)
            for i, comp_mask in enumerate(component_masks):
                component_masks[i] = binary_dilation(comp_mask, iterations=dilation_iters)
    # Replace the anomaly with gaussian sky noise
    if replace_anomaly:
        bkg_rms = estimate_background(image)
        # Generate gaussian noise based on the estimated background rms
        gaussian_noise = np.random.normal(0, bkg_rms, np.count_nonzero(global_mask))
        # Modify the original image using the accumulated total mask
        image_mod = image.copy()
        image_mod[global_mask] = gaussian_noise

    if show_plot:
        plt.figure(figsize=(8, 8), constrained_layout=True)
        # Original image
        plt.subplot(231)
        norm = simple_norm(image, 'sqrt', percent=96.0)
        plt.imshow(image, cmap='gray_r', origin='lower', norm=norm)  # type: ignore
        plt.title('Original Image')
        plt.axis('off')
        # Modified image
        plt.subplot(232)
        if not replace_anomaly:
            image_mod = image
        norm_mod = simple_norm(image_mod, 'sqrt', percent=96.0)
        plt.imshow(image_mod, cmap='gray_r', origin='lower', norm=norm_mod)  # type: ignore
        if np.any(global_mask):
            plt.imshow(global_mask, cmap='Reds', vmin=0, alpha=0.5, origin='lower')
        plt.title('Modified Image')
        plt.axis('off')
        # Diagonal coefficients
        plt.subplot(234)
        norm_cD = simple_norm(cD, 'sqrt', percent=96.0)
        plt.imshow(cD, cmap='gray_r', origin='lower', norm=norm_cD)  # type: ignore
        if np.any(component_masks[0] > 0.0):
            plt.imshow(
                component_masks[0], cmap='Reds', vmin=0, alpha=0.5, origin='lower'
            )  # Overlay connected components
        plt.title('Diagonal Coefficients')
        plt.axis('off')
        # Horizontal coefficients
        plt.subplot(235)
        norm_cH = simple_norm(cH, 'sqrt', percent=96.0)
        plt.imshow(cH, cmap='gray_r', origin='lower', norm=norm_cH)  # type: ignore
        if np.any(component_masks[1] > 0.0):
            plt.imshow(
                component_masks[1], cmap='Reds', vmin=0, alpha=0.5, origin='lower'
            )  # Overlay connected components
        plt.title('Horizontal Coefficients')
        plt.axis('off')
        # Vertical coefficients
        plt.subplot(236)
        norm_cV = simple_norm(cV, 'sqrt', percent=96.0)
        plt.imshow(cV, cmap='gray_r', origin='lower', norm=norm_cV)  # type: ignore
        if np.any(component_masks[2] > 0.0):
            plt.imshow(
                component_masks[2], cmap='Reds', vmin=0, alpha=0.5, origin='lower'
            )  # Overlay connected components
        plt.title('Vertical Coefficients')
        plt.axis('off')
        plt.show()

        if save_plot:
            plt.savefig(
                os.path.join(figure_path, f'{file_out}.pdf'),
                bbox_inches='tight',
                dpi=300,
            )
        else:
            plt.close()

    return anomalies

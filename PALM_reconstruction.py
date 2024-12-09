"""
This script processes localization data from a CSV file to generate a PALM (Photo-Activated Localization Microscopy) 
reconstruction. It includes functionality to simulate the spatial distribution of molecules using Gaussian functions 
based on localization uncertainty values.

The script performs the following:
1. Reads x, y coordinates and localization uncertainties from a CSV file.
2. Converts the coordinates and uncertainties from nanometers to micrometers.
3. Reconstructs a high-resolution image using the PALM technique by overlaying Gaussian distributions at each 
   localization point.
4. Visualizes the reconstructed image using a heatmap.

Functions:
- draw_gaussian: Adds a Gaussian distribution centered at a specified point to an image.
- palm: Reconstructs a high-resolution PALM image based on the localization data.

Authors:
- D. Linnik (d.linnik at rug.nl)
- C.M. Punter (c.m.punter at rug.nl)
Membrane Enzymology, University of Groningen, 2024.
"""

import numpy as np
import matplotlib.pyplot as plt

path = "Path_to_csv_file"

data = np.genfromtxt(path,  delimiter=',')

# get the xy coordinates and localization uncertainty values and convert them from nm to um
x = data[1:, 1] / 1000
y = data[1:, 2] / 1000
uncertainty = data[1:, 3] / 1000


def draw_gaussian(im, x, y, sigma):
    """
    Draws a normal distribution centered around (x, y) with a given sigma on the image.
    Adds the values of the normal distribution to the pre-existing values in the image.

    Args:
        im (ndarray): Input image.
        x (float): x-coordinate of the center.
        y (float): y-coordinate of the center.
        sigma (float): Standard deviation of the normal distribution.

    Returns:
        ndarray: Image with the added values of the normal distribution.
    """

    # Generate a grid of coordinates
    height, width = im.shape

    xmin = max(0, int(x - sigma * 3))
    xmax = min(width, int(x + sigma * 3))
    ymin = max(0, int(y - sigma * 3))
    ymax = min(height, int(y + sigma * 3))

    x_grid, y_grid = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # Calculate the distances from the center
    distances = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

    # Compute the normal distribution
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distances / sigma) ** 2)

    # Add the values of the normal distribution to the image
    im[ymin:ymax, xmin:xmax] += gaussian

    return im


def palm(x, y, e, xmin, xmax, ymin, ymax, pixel_size):

    # Reconstructed image size determination
    width = int((xmax - xmin) / pixel_size) + 1
    height = int((ymax - ymin) / pixel_size) + 1
    reconstruction = np.zeros((height, width))

    for x1, y1, e1 in zip(x, y, e):
        draw_gaussian(reconstruction, (x1 - xmin) / pixel_size, (y1 - ymin) / pixel_size, e1 / pixel_size)

    return reconstruction

# Setting the magnification value
magnification = 100
sigma = uncertainty / magnification

reconstruction_global = palm(x, y, uncertainty, np.min(x), np.max(x), np.min(y), np.max(y), 1 / magnification)

fig, ax = plt.subplots(figsize=(30, 10))
ax.set_xlabel('X, um')
ax.set_ylabel('Y, um')
ax.set_title('PALM reconstruction')
ax.imshow(reconstruction_global, origin='lower', cmap="hot")

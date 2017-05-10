import numpy as np
import cv2
from skimage.feature import hog

class FeatureExtractor(object):

    def __init__(self, image, num_orientations=10, pixels_per_cell=8, cells_per_block=2):
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        num_channels = self.image.shape[2]
        self.pixels_per_cell = pixels_per_cell
        self.hog_features = np.array([
            hog(
                self.image[:, :, channel], 
                orientations=num_orientations, 
                pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block), 
                transform_sqrt=True,
                visualise=False, 
                feature_vector=False
            ) for channel in range(num_channels)
        ])

    def hog(self, x, y, window_size):
        ppc = self.pixels_per_cell

        x_size = self.hog_features.shape[2]
        y_size = self.hog_features.shape[1]

        hog_window = (window_size // ppc) - 1
        hog_x = max((x // ppc) - 1, 0)
        hog_x = x_size - hog_window if hog_x + hog_window > x_size else hog_x
        hog_y = max((y // ppc) - 1, 0)
        hog_y = y_size - hog_window if hog_y + hog_window > y_size else hog_y

        return np.ravel(
            self.hog_features[
                :,
                hog_y : hog_y + hog_window,
                hog_x : hog_x + hog_window,
                :, :, :
            ]
        )

    def bin_spatial(self, image, size=(16, 16)):
        return cv2.resize(image, size).ravel()

    # Define a function to compute color histogram features
    def color_histogram(self, image, number_of_bins=16, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(image[:,:,0], bins=number_of_bins, range=bins_range)
        channel2_hist = np.histogram(image[:,:,1], bins=number_of_bins, range=bins_range)
        channel3_hist = np.histogram(image[:,:,2], bins=number_of_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    def feature_vector(self, x=0, y=0, window_size=64):
        features = []

        spatial_features = self.bin_spatial(
            self.image[
                y : y + window_size,
                x : x + window_size,
                :
            ]
        )
        features.append(spatial_features)

        color_histogram_features = self.color_histogram(
            self.image[
                y : y + window_size,
                x : x + window_size,
                :
            ]
        )
        features.append(color_histogram_features)

        hog_features = self.hog(x, y, window_size)
        features.append(hog_features)

        return np.concatenate(features)
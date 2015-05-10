"""

Module Name: TSEResult

Description: Provides compiled (C code) versions of certain functions originally available within
'TSEImageUtils', which are known to be computationally expensive.

Note throughout this module the use of 'Cython-specific' syntax required to fully translate Python to C code.

See: http://docs.cython.org/src/userguide/language_basics.html for more information.

"""

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'

import numpy as np
cimport numpy as np

cdef class TSECImageUtils:

    @staticmethod
    # For full translation to C, we must provide the parameter and variable types.
    def calc_ssd_slow(np.ndarray[np.uint8_t, ndim=3] template_patch, np.ndarray[np.uint8_t, ndim=3] scaled_current_window, int template_patch_height, int template_patch_width, float scale_factor_height, float scale_factor_width):
        """
        Calculates the Sum of Squared Difference value between two images using a non-efficient multi-nested loop approach.

        :param template_patch: The template image to be compared.
        :param scaled_current_window: The larger search window to also be compared.
        :param template_patch_height: Height of template patch.
        :param template_patch_width: Width of template patch.
        :param scale_factor_height: Scale factor between height of template patch and larger search window.
        :param scale_factor_width: Scale factor between height of template patch and larger search window.
        :return Float representing the match score calculated by the SSD metric.
        """

        # In Cython, we must define the types for all of our local variables.
        cdef unsigned int i, j, template_patch_val_channel_1, template_patch_val_channel_2, template_patch_val_channel_3, scaled_current_window_val_channel_1, scaled_current_window_val_channel_2, scaled_current_window_val_channel_3, diff_channel_1, diff_channel_2, diff_channel_3
        cdef float ssd = 0

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(template_patch_height):
            for j in xrange(template_patch_width):

                # Extract each of the three channels for each pixel separately.
                template_patch_val_channel_1 = template_patch[i][j][0]
                template_patch_val_channel_2 = template_patch[i][j][1]
                template_patch_val_channel_3 = template_patch[i][j][2]

                # Calculate the scaled coordinate of the original pixel and extract the three channels of the corresponding pixel from within the larger image.
                scaled_current_window_val_channel_1 = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][0]
                scaled_current_window_val_channel_2 = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][1]
                scaled_current_window_val_channel_3 = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][2]

                # Calculate the Sum of Squared Difference between each of the channels independently.
                diff_channel_1 = template_patch_val_channel_1 - scaled_current_window_val_channel_1
                diff_channel_2 = template_patch_val_channel_2 - scaled_current_window_val_channel_2
                diff_channel_3 = template_patch_val_channel_3 - scaled_current_window_val_channel_3

                # For some reason the line below doesn't work.. we have to add each squared value separately.
                # ssd += (diff_channel_1 * diff_channel_1) + (diff_channel_2 * diff_channel_2) + (diff_channel_3 * diff_channel_3)

                ssd += (diff_channel_1 * diff_channel_1)
                ssd += (diff_channel_2 * diff_channel_2)
                ssd += (diff_channel_3 * diff_channel_3)

        return ssd

    @staticmethod
    def extract_rows_cols_pixels_image(np.ndarray[np.float64_t, ndim=1] required_rows, np.ndarray[np.float64_t, ndim=1] required_cols, np.ndarray[np.uint8_t, ndim=3] image):
        """
        Extracts the pixel defined at each specified row-column position within the image by calculating the cartesian product
        between the collection of specified rows and columns.

        :param required_rows: The collection of required image rows.
        :param required_rows: The collection of required image columns.
        :param image: The image from which the specified pixels are to be extracted.
        :return The subset of pixels from the original image as defined by the cartesian product
        between the collection of specified rows and columns.
        """

        # Get the cartesian product between the two then split into one array for all rows, and one array for all cols.
        cdef list rows_cols_cartesian_product = np.hsplit(TSECImageUtils.calc_cartesian_product([required_rows, required_cols]), 2)

        cdef np.ndarray[np.int64_t, ndim=2] rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cdef np.ndarray[np.int64_t, ndim=2] cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        return image[rows_to_extract, cols_to_extract]

    @staticmethod
    # This code has been modified from an original source: http://stackoverflow.com/a/11146645/4768230
    def calc_cartesian_product(list arrays):
        """
        Calculates the cartesian product of elements between a collection of 1D arrays.

        :param arrays: The collection of 1D arrays of which each element will be combined with each other.
        :return: The combined results of the cartesian product between all input elements.
        """

        cdef int i, la = len(arrays)

        cdef np.ndarray arr = np.empty([len(a) for a in arrays] + [la])

        cdef np.ndarray a

        # Perform cross combination across each array in turn.
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a

        return arr.reshape(-1, la)

    @staticmethod
    def reshape_match_images(np.ndarray[np.uint8_t, ndim=3] current_matrix, np.ndarray[np.uint8_t, ndim=3] target_matrix):
        """
        If required, re-shapes the structure of two Numpy 'ndarray' objects representing images so that they match.

        :param current_matrix: The matrix object to be re-shaped.
        :param target_matrix: The matric object of whose shape is to be matched.
        :return The same Numpy 'ndarray' object, but with potentially a different internal dimension structure.
        """

        cdef int height = target_matrix.shape[0]
        cdef int width = target_matrix.shape[1]
        cdef int depth = target_matrix.shape[2]

        if current_matrix.shape != target_matrix.shape:
            return current_matrix.reshape(height, width, depth)

        return current_matrix
__author__ = 'connorgoddard'

import numpy as np
cimport numpy as np

cdef class TSECImageUtils:

    @staticmethod
    def calc_ssd_slow(np.ndarray[np.uint8_t, ndim=3] template_patch, np.ndarray[np.uint8_t, ndim=3] scaled_current_window, int template_patch_height, int template_patch_width, float scale_factor_height, float scale_factor_width):

        cdef unsigned int i, j, template_patch_val_channel_1, template_patch_val_channel_2, template_patch_val_channel_3, scaled_current_window_val_channel_1, scaled_current_window_val_channel_2, scaled_current_window_val_channel_3, diff_channel_1, diff_channel_2, diff_channel_3
        cdef float ssd = 0

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(template_patch_height):
            for j in xrange(template_patch_width):

                template_patch_val_channel_1 = template_patch[i][j][0]
                template_patch_val_channel_2 = template_patch[i][j][1]
                template_patch_val_channel_3 = template_patch[i][j][2]

                scaled_current_window_val_channel_1 = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][0]
                scaled_current_window_val_channel_2 = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][1]
                scaled_current_window_val_channel_3 = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][2]

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

        # Get the cartesian product between the two then split into one array for all rows, and one array for all cols.
        cdef list rows_cols_cartesian_product = np.hsplit(TSECImageUtils.calc_cartesian_product([required_rows, required_cols]), 2)

        cdef np.ndarray[np.int64_t, ndim=2] rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cdef np.ndarray[np.int64_t, ndim=2] cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        return image[rows_to_extract, cols_to_extract]

    @staticmethod
    def calc_cartesian_product(list arrays):

        cdef int i, la = len(arrays)

        cdef np.ndarray arr = np.empty([len(a) for a in arrays] + [la])

        cdef np.ndarray a

        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a

        return arr.reshape(-1, la)

    @staticmethod
    def reshape_match_images(np.ndarray[np.uint8_t, ndim=3] current_matrix, np.ndarray[np.uint8_t, ndim=3] target_matrix):

        cdef int height = target_matrix.shape[0]
        cdef int width = target_matrix.shape[1]
        cdef int depth = target_matrix.shape[2]

        if current_matrix.shape != target_matrix.shape:
            return current_matrix.reshape(height, width, depth)

        return current_matrix
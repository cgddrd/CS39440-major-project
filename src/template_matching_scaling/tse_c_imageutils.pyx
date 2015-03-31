import numpy as np
cimport numpy as np

def calc_ssd_compiled(np.ndarray[np.uint8_t, ndim=3] template_patch, np.ndarray[np.uint8_t, ndim=3] scaled_current_window, int template_patch_height, int template_patch_width, float scale_factor_height, float scale_factor_width):

    cdef unsigned int i, j, template_patch_val_hue, template_patch_val_sat, scaled_current_window_val_hue, scaled_current_window_val_sat, diff_hue, diff_sat
    cdef float ssd = 0

    # Loop through each pixel in the template patch, and scale it in the larger scaled image.
    for i in xrange(template_patch_height):
        for j in xrange(template_patch_width):

            template_patch_val_hue = template_patch[i][j][0]
            template_patch_val_sat = template_patch[i][j][1]

            scaled_current_window_val_hue = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][0]
            scaled_current_window_val_sat = scaled_current_window[(i * scale_factor_height)][(j * scale_factor_width)][1]

            # print scaled_current_window_val_hue

            diff_hue = template_patch_val_hue - scaled_current_window_val_hue
            diff_sat = template_patch_val_sat - scaled_current_window_val_sat


            ssd += ((diff_hue * diff_hue) + (diff_sat * diff_sat))

    return ssd

def extract_rows_cols_image(np.ndarray[np.float64_t, ndim=1] required_rows, np.ndarray[np.float64_t, ndim=1] required_cols, np.ndarray[np.uint8_t, ndim=3] image):

    # Get the cartesian product between the two then split into one array for all rows, and one array for all cols.
    cdef list rows_cols_cartesian_product = np.hsplit(calc_cartesian_product([required_rows, required_cols]), 2)

    cdef np.ndarray[np.int64_t, ndim=2] rows_to_extract = rows_cols_cartesian_product[0].astype(int)
    cdef np.ndarray[np.int64_t, ndim=2] cols_to_extract = rows_cols_cartesian_product[1].astype(int)

    return image[rows_to_extract, cols_to_extract]

def calc_cartesian_product(list arrays):

    cdef int i, la = len(arrays)

    cdef np.ndarray arr = np.empty([len(a) for a in arrays] + [la])

    cdef np.ndarray a

    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a

    return arr.reshape(-1, la)

def reshape_image_matrix(np.ndarray[np.uint8_t, ndim=3] current_matrix, np.ndarray[np.uint8_t, ndim=3] target_matrix):

    cdef int height = target_matrix.shape[0]
    cdef int width = target_matrix.shape[1]
    cdef int depth = target_matrix.shape[2]

    if current_matrix.shape != target_matrix.shape:
        return current_matrix.reshape(height, width, depth)

    return current_matrix
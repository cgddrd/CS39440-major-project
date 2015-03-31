__author__ = 'connorgoddard'

import math
import test2

import cv2
import numpy as np

from tse.tse_geometry import TSEGeometry
from tse_utils import TSEUtils


class TSEImageUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def blah(template_patch, scaled_current_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_current_window_height, scaled_current_window_width = scaled_current_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_patch_scale_factor(template_patch_width, scaled_current_window_width)
        scale_factor_height = TSEGeometry.calc_patch_scale_factor(template_patch_height, scaled_current_window_height)

        ssd = test2.calc_ssd_compiled(template_patch, scaled_current_window, template_patch_height, template_patch_width,
                                     scale_factor_height, scale_factor_width)

        return math.sqrt(ssd)

    @staticmethod
    def blah2(template_patch, scaled_current_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_current_window_height, scaled_current_window_width = scaled_current_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_patch_scale_factor(template_patch_width, scaled_current_window_width)
        scale_factor_height = TSEGeometry.calc_patch_scale_factor(template_patch_height, scaled_current_window_height)

        template_patch_heights = np.arange(0, template_patch_height)
        template_patch_widths = np.arange(0, template_patch_width)

        scaled_window_heights = np.rint(template_patch_heights * scale_factor_height)
        scaled_window_widths = np.rint(template_patch_widths * scale_factor_width)

        # result = TSEImageUtils.extract_rows_cols_image(scaled_window_heights, scaled_window_widths,scaled_current_window)

        result = test2.extract_rows_cols_image(scaled_window_heights, scaled_window_widths, scaled_current_window)

        reshaped_result = TSEImageUtils.reshape_image_matrix(result, template_patch)

        return TSEImageUtils.calc_euclidean_distance_norm(template_patch, reshaped_result)

    @staticmethod
    def blah3(template_patch, scaled_current_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_current_window_height, scaled_current_window_width = scaled_current_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_patch_scale_factor(template_patch_width, scaled_current_window_width)
        scale_factor_height = TSEGeometry.calc_patch_scale_factor(template_patch_height, scaled_current_window_height)

        template_patch_heights = np.arange(0, template_patch_height)
        template_patch_widths = np.arange(0, template_patch_width)

        scaled_window_heights = np.rint(template_patch_heights * scale_factor_height)
        scaled_window_widths = np.rint(template_patch_widths * scale_factor_width)

        result = test2.extract_rows_cols_image(scaled_window_heights, scaled_window_widths, scaled_current_window)

        reshaped_result = test2.reshape_image_matrix(result, template_patch)

        return TSEImageUtils.calc_euclidean_distance_norm(template_patch, reshaped_result)

    @staticmethod
    def calc_euclidean_distance_norm(patch1, patch2):
        return cv2.norm(patch1, patch2, cv2.NORM_L2)

    @staticmethod
    def reshape_image_matrix(current_matrix, target_matrix):

        if current_matrix.shape != target_matrix.shape:
            return current_matrix.reshape(target_matrix.shape)

        return current_matrix

    @staticmethod
    def extract_rows_cols_image(required_rows, required_cols, image):

        # Get the cartesian product between the two then split into one array for all rows, and one array for all cols.
        rows_cols_cartesian_product = np.hsplit(TSEUtils.calc_cartesian_product([required_rows, required_cols]), 2)

        rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        return image[rows_to_extract, cols_to_extract]


    @staticmethod
    def calc_euclidean_distance_scaled(template_patch, scaled_current_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_current_window_height, scaled_current_window_width = scaled_current_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_patch_scale_factor(template_patch_width, scaled_current_window_width)
        scale_factor_height = TSEGeometry.calc_patch_scale_factor(template_patch_height, scaled_current_window_height)

        ssd = 0

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(template_patch_height):
            for j in xrange(template_patch_width):
                # WE DON'T BOTHER DOING ANYTHING WITH THE 'V' CHANNEL, AS WE ARE IGNORING IT ANYWAY.
                template_patch_val_hue = template_patch.item(i, j, 0)
                template_patch_val_sat = template_patch.item(i, j, 1)

                scaled_current_window_val_hue = scaled_current_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 0)
                scaled_current_window_val_sat = scaled_current_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 1)

                diff_hue = template_patch_val_hue - scaled_current_window_val_hue

                diff_sat = template_patch_val_sat - scaled_current_window_val_sat

                ssd += (diff_hue * diff_hue)

                ssd += (diff_sat * diff_sat)

        return math.sqrt(ssd)

    @staticmethod
    def calc_cross_correlation_normed(patch1, patch2):

        res = cv2.matchTemplate(patch2, patch1, cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        return max_val

    @staticmethod
    def calc_correlation_coeff_normed(patch1, patch2):

        res = cv2.matchTemplate(patch2, patch1, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        return max_val

    @staticmethod
    def calc_match_compare(patch1, patch2, match_method):

        res = cv2.matchTemplate(patch2, patch1, match_method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        return max_val

    @staticmethod
    def hist_compare(patch1, patch2, match_method):

        hist_patch1 = cv2.calcHist([patch1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_patch2 = cv2.calcHist([patch2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        hist_compare_result = cv2.compareHist(hist_patch1, hist_patch2, match_method)

        return hist_compare_result

    @staticmethod
    def convert_hsv_and_remove_luminance(source_image):
        hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)

        # Set the 'V' channel of each pixel in the image to '0' (i.e remove it)
        # hsv_image[:, :, 1] = 0
        hsv_image[:, :, 2] = 0

        return hsv_image

    @staticmethod
    def scale_image_roi_relative_centre(origin, end, scale_factor):

        height = end[1] - origin[1]
        width = end[0] - origin[0]

        centre = ((origin[0] + (width / 2)), (origin[1] + (height / 2)))

        scaled_origin = TSEGeometry.scale_coordinate_relative_centre(origin, centre, scale_factor)

        scaled_end = TSEGeometry.scale_coordinate_relative_centre(end, centre, scale_factor)

        return scaled_origin, scaled_end

    @staticmethod
    # METHOD ASSUMES WE ARE DEALING WITH HSV IMAGES WITH 'V' CHANNEL REMOVED.
    def scale_image_no_interpolation(source_image, current_image, scale_factor):

        source_image_height, source_image_width = source_image.shape[:2]

        current_image_height, current_image_width = current_image.shape[:2]

        scale_factor_height = TSEGeometry.calc_patch_scale_factor(source_image_height, current_image_height)
        scale_factor_width = TSEGeometry.calc_patch_scale_factor(source_image_width, current_image_width)

        scaled_source_image_height = round(source_image_height * scale_factor_height)
        scaled_source_image_width = round(source_image_width * scale_factor_width)

        scaled_image_result = np.zeros((scaled_source_image_height, scaled_source_image_width, 3), np.uint8)

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(source_image_height):
            for j in xrange(source_image_width):
                # WE DON'T BOTHER DOING ANYTHING WITH THE 'V' CHANNEL, AS WE ARE IGNORING IT ANYWAY.
                template_patch_val_hue = source_image.item(i, j, 0)
                template_patch_val_sat = source_image.item(i, j, 1)

                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 0),
                                            template_patch_val_hue)
                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 1),
                                            template_patch_val_sat)

        return scaled_image_result

    @staticmethod
    def scale_image_interpolation(source_image, scale_factor):

        source_image_height, source_image_width = source_image.shape[:2]

        dim = (int(source_image_width * scale_factor), int(source_image_height * scale_factor))

        return cv2.resize(source_image, dim)
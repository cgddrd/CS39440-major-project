from __future__ import division

import math

import cv2
import numpy as np

from tse_compiled.tse_c_imageutils import TSECImageUtils
from tse.tse_geometry import TSEGeometry
from tse_utils import TSEUtils
from tse_point import TSEPoint


__author__ = 'connorgoddard'


class TSEImageUtils:
    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_euclidean_distance_cv2_norm(image_1, image_2):
        return cv2.norm(image_1, image_2, cv2.NORM_L2)

    @staticmethod
    def calc_ed_template_match_score_scaled(template_patch, scaled_search_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, scaled_search_window_width)
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, scaled_search_window_height)

        scaled_window_heights = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_height, scale_factor_height)
        scaled_window_widths = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_width, scale_factor_width)

        search_window_target_pixels = TSEImageUtils.extract_rows_cols_pixels_image(scaled_window_heights, scaled_window_widths, scaled_search_window)

        reshaped_search_window_target_pixels = TSEImageUtils.reshape_match_images(search_window_target_pixels, template_patch)

        return TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, reshaped_search_window_target_pixels)

    @staticmethod
    def calc_ed_template_match_score_scaled_slow(template_patch, scaled_search_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, scaled_search_window_width)
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, scaled_search_window_height)

        ssd = 0

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(template_patch_height):
            for j in xrange(template_patch_width):
                # WE DON'T BOTHER DOING ANYTHING WITH THE 'V' CHANNEL, AS WE ARE IGNORING IT ANYWAY.
                template_patch_val_hue = template_patch.item(i, j, 0)
                template_patch_val_sat = template_patch.item(i, j, 1)

                scaled_search_window_val_hue = scaled_search_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 0)
                scaled_search_window_val_sat = scaled_search_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 1)

                diff_hue = template_patch_val_hue - scaled_search_window_val_hue

                diff_sat = template_patch_val_sat - scaled_search_window_val_sat

                ssd += (diff_hue * diff_hue)

                ssd += (diff_sat * diff_sat)

        return math.sqrt(ssd)

    @staticmethod
    def calc_ed_template_match_score_scaled_compiled(template_patch, scaled_search_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, scaled_search_window_width)
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, scaled_search_window_height)

        scaled_window_heights = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_height, scale_factor_height)
        scaled_window_widths = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_width, scale_factor_width)

        search_window_target_pixels = TSECImageUtils.extract_rows_cols_pixels_image(scaled_window_heights, scaled_window_widths, scaled_search_window)

        reshaped_search_window_target_pixels = TSECImageUtils.reshape_match_images(search_window_target_pixels, template_patch)

        return TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, reshaped_search_window_target_pixels)

    @staticmethod
    def calc_ed_template_match_score_scaled_compiled_slow(template_patch, scaled_search_window):

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, scaled_search_window_width)
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, scaled_search_window_height)

        ssd = TSECImageUtils.calc_ssd_slow(template_patch, scaled_search_window, template_patch_height, template_patch_width, scale_factor_height, scale_factor_width)

        return math.sqrt(ssd)

    @staticmethod
    def calc_scaled_image_pixel_dimension_coordinates(image_dim_end, scale_factor, image_dim_start=0, round=True):

        image_dim_coordinates = np.arange(image_dim_start, image_dim_end)

        # Perform Numpy element-wise multiplication function to all elements in array.
        scaled_image_dim_coordinates = image_dim_coordinates * scale_factor

        # As we are dealing in image pixels, we will normally want to round the results to the nearest integer.
        if round is True:
            return np.rint(scaled_image_dim_coordinates)

        return scaled_image_dim_coordinates

    @staticmethod
    def reshape_match_images(current_matrix, target_matrix):

        if current_matrix.shape != target_matrix.shape:
            return current_matrix.reshape(target_matrix.shape)

        return current_matrix

    @staticmethod
    def extract_rows_cols_pixels_image(required_rows, required_cols, image):

        # Get the cartesian product between the two then split into one array for all rows, and one array for all cols.
        rows_cols_cartesian_product = np.hsplit(TSEUtils.calc_cartesian_product([required_rows, required_cols]), 2)

        rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        return image[rows_to_extract, cols_to_extract]

    @staticmethod
    def calc_template_match_compare_cv2_score(image_1, image_2, match_method):

        res = cv2.matchTemplate(image_2, image_1, match_method)

        # '_' is a placeholder convention to indicate we do not want to use these returned values.
        min_val, max_val, _, _ = cv2.minMaxLoc(res)

        # If we are matching using SSD, then the lowest score is the "best match", so return this.
        if match_method == cv2.cv.CV_TM_SQDIFF or match_method == cv2.cv.CV_TM_SQDIFF_NORMED:
            return min_val

        # Otherwise (and in most cases), we will want to return the highest score.
        return max_val

    @staticmethod
    def calc_template_match_compare_cv2_score_scaled(image_1, image_2, match_method):

        template_patch_height, template_patch_width = image_1.shape[:2]
        scaled_search_window_height, scaled_search_window_width = image_2.shape[:2]

        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, scaled_search_window_width)
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, scaled_search_window_height)

        scaled_window_heights = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_height, scale_factor_height)
        scaled_window_widths = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_width, scale_factor_width)

        search_window_target_pixels = TSEImageUtils.extract_rows_cols_pixels_image(scaled_window_heights, scaled_window_widths, image_2)

        reshaped_search_window_target_pixels = TSEImageUtils.reshape_match_images(search_window_target_pixels, image_1)

        res = cv2.matchTemplate(reshaped_search_window_target_pixels, image_1, match_method)

        # '_' is a placeholder convention to indicate we do not want to use these returned values.
        min_val, max_val, _, _ = cv2.minMaxLoc(res)

        # If we are matching using SSD, then the lowest score is the "best match", so return this.
        if match_method == cv2.cv.CV_TM_SQDIFF or match_method == cv2.cv.CV_TM_SQDIFF_NORMED:
            return min_val

        # Otherwise (and in most cases), we will want to return the highest score.
        return max_val


    @staticmethod
    def calc_compare_histogram(image_1, image_2, match_method):

        hist_patch1 = cv2.calcHist([image_1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_patch2 = cv2.calcHist([image_2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        hist_compare_result = cv2.compareHist(hist_patch1, hist_patch2, match_method)

        return hist_compare_result

    @staticmethod
    def convert_hsv_and_remove_luminance(image):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Set the 'V' channel of each pixel in the image to '0' (i.e remove it)
        hsv_image[:, :, 2] = 0

        return hsv_image

    @staticmethod
    def scale_image_roi_relative_centre(origin_coordinate, end_coordinate, scale_factor):

        height = end_coordinate[1] - origin_coordinate[1]
        width = end_coordinate[0] - origin_coordinate[0]

        centre = ((origin_coordinate[0] + (width / 2)), (origin_coordinate[1] + (height / 2)))

        scaled_origin = TSEGeometry.scale_coordinate_relative_centre(origin_coordinate, centre, scale_factor)

        scaled_end = TSEGeometry.scale_coordinate_relative_centre(end_coordinate, centre, scale_factor)

        return TSEPoint(scaled_origin[0], scaled_origin[1]), TSEPoint(scaled_end[0], scaled_end[1])

    @staticmethod
    # METHOD ASSUMES WE ARE DEALING WITH HSV IMAGES WITH 'V' CHANNEL REMOVED.
    def scale_image_no_interpolation_auto(source_image, target_image):

        source_image_height, source_image_width = source_image.shape[:2]
        current_image_height, current_image_width = target_image.shape[:2]

        scale_factor_height = TSEGeometry.calc_measure_scale_factor(source_image_height, current_image_height)
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(source_image_width, current_image_width)

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
    def scale_image_interpolation_auto(source_image, target_image):

        source_image_height, source_image_width = source_image.shape[:2]
        current_image_height, current_image_width = target_image.shape[:2]

        scale_factor_height = TSEGeometry.calc_measure_scale_factor(source_image_height, current_image_height)
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(source_image_width, current_image_width)

        scaled_source_image_height = round(source_image_height * scale_factor_height)
        scaled_source_image_width = round(source_image_width * scale_factor_width)

        dim = (int(scaled_source_image_width), int(scaled_source_image_height))

        return cv2.resize(source_image, dim)

    @staticmethod
    def scale_image_interpolation_man(source_image, scale_factor):

        source_image_height, source_image_width = source_image.shape[:2]

        dim = (int(source_image_width * scale_factor), int(source_image_height * scale_factor))

        return cv2.resize(source_image, dim)

    @staticmethod
    def extract_image_sub_window(source_image, origin_coordinates, end_coordinates):
        return source_image[origin_coordinates.y:end_coordinates.y, origin_coordinates.x: end_coordinates.x]

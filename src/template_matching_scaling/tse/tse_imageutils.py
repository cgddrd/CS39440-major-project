__author__ = 'connorgoddard'

import cv2
import numpy as np
import math
from tse.tse_geometry import TSEGeometry


class TSEImageUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_euclidean_distance_norm(patch1, patch2):

        # cv2.imshow("patch1", patch1)
        # cv2.imshow("patch2", patch2)
        #
        print "Scaled Template Image: {0}".format(patch1.shape)
        print "Current Window: {0}".format(patch2.shape)

        # print "\n"

        # print patch1.type()
        # print patch2.type()

        test = cv2.norm(patch1, patch2, cv2.NORM_L2)

        # print test

        # cv2.waitKey()

        return test

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

        # print "Scale Factor: {0}".format(scale_factor)
        # print "TARGET Height: {0}".format(source_image_height * scale_factor)
        # print "TARGET Width: {0}".format(source_image_width * scale_factor)

        scaled_source_image_height = round(source_image_height * scale_factor_height)
        scaled_source_image_width = round(source_image_width * scale_factor_width)

        # print "ACTUAL HEIGHT: {0}".format(scaled_source_image_height)
        # print "ACTUAL WIDTH: {0}".format(scaled_source_image_width)

        scaled_image_result = np.zeros((scaled_source_image_height, scaled_source_image_width, 3), np.uint8)

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(source_image_height):
            for j in xrange(source_image_width):

                # WE DON'T BOTHER DOING ANYTHING WITH THE 'V' CHANNEL, AS WE ARE IGNORING IT ANYWAY.
                template_patch_val_hue = source_image.item(i, j, 0)
                template_patch_val_sat = source_image.item(i, j, 1)

                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 0), template_patch_val_hue)
                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 1), template_patch_val_sat)

        return scaled_image_result

    @staticmethod
    def scale_image_interpolation(source_image, scale_factor):

        source_image_height, source_image_width = source_image.shape[:2]

        dim = (int(source_image_width * scale_factor), int(source_image_height * scale_factor))

        return cv2.resize(source_image, dim)
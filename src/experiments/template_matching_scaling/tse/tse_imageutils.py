__author__ = 'connorgoddard'

import cv2


class TSEImageUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_euclidean_distance_norm(patch1, patch2):
        return cv2.norm(patch1, patch2, cv2.NORM_L2)

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
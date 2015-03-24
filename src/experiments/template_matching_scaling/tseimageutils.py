__author__ = 'connorgoddard'

import cv2


class TSEImageUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_euclidean_distance_norm(patch1, patch2):

        return cv2.norm(patch1, patch2, cv2.NORM_L2)
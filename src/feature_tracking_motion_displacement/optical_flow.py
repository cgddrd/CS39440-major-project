__author__ = 'connorgoddard'

import cv2

from optical_flow_settings import *

class OpticalFlow:

    def __init__(self, feature_detector_config, subpix_feature_detector_config, feature_tracker_config):
        self._feature_detector_config = feature_detector_config
        self._subpix_feature_detector_config = subpix_feature_detector_config
        self._feature_tracker_config = feature_tracker_config

    @classmethod
    def load_defaults(cls):
        return cls(feature_detector_defaults, subpix_feature_detector_defaults, feature_tracker_defaults)

    def find_features_to_track(self, input_image_gray):

        points = cv2.goodFeaturesToTrack(input_image_gray, **self._feature_detector_config)

        cv2.cornerSubPix(input_image_gray, points, **self._subpix_feature_detector_config)

        return points

    def calc_optical_flow(self, prev_gray_image, current_gray_image, prev_feature_points):

        return cv2.calcOpticalFlowPyrLK(prev_gray_image, current_gray_image, prev_feature_points, None, **self._feature_tracker_config)
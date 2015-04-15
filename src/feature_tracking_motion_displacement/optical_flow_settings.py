import cv2

__author__ = 'connorgoddard'

# optical_flow_defaults = {'max_feature_count': 10,
#                          'block_size': 5,
#                          'quality_level': 0.01,
#                          'min_distance': 10}

feature_detector_defaults = dict(maxCorners=100, qualityLevel=0.01, minDistance=20)

subpix_feature_detector_defaults = dict(winSize=(10, 10), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

feature_tracker_defaults = dict(winSize=(20, 20), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.3))
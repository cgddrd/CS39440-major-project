from __future__ import division

__author__ = 'connorgoddard'

import cv2
import numpy as np

from optical_flow import OpticalFlow
from tracked_feature import TrackedFeature
# from circular_buffer import CircularBuffer


class VisualOdometer:

    _optical_flow = OpticalFlow.load_defaults()
    _current_gray_image = None
    _prev_gray_image = None
    _raw_tracked_feature_points = []
    _tracked_features = []
    _initial_features_count = 0
    _feature_repopulation_threshold = 10
    _not_tracked_feature_count = 0

    def __init__(self):
        pass

    def process_image(self, new_image_filepath):

        self._prev_gray_image = self._current_gray_image

        new_image = cv2.imread(new_image_filepath)
        self._current_gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        if self._prev_gray_image is None:
            self.repopulate_feature_points(self._current_gray_image)
            return

        # cv2.imshow("prev", self._prev_gray_image)
        # cv2.imshow("current", self._current_gray_image)

        self.track_features()

        if len(self._tracked_features) < self._feature_repopulation_threshold:

            print "We need to re-populate features!"
            self.repopulate_feature_points(self._current_gray_image)

    def track_features(self):

        if len(self._raw_tracked_feature_points) == 0:
            return

        tracked_feature_points, tracking_status_indicators, tracking_errors = self._optical_flow.calc_optical_flow(self._prev_gray_image, self._current_gray_image, np.array(self._raw_tracked_feature_points))

        # Clear the current list of tracked feature points ready to replace with the new ones.
        self._raw_tracked_feature_points = []

        self._raw_tracked_feature_points.extend(tracked_feature_points)

        full_history_features_count = 0
        unsmooth_features_count = 0

        for i in reversed(xrange(len(tracked_feature_points))):

            is_tracked = (tracking_status_indicators[i] == 1)

            if is_tracked:

                existing_tracked_feature = self._tracked_features[i]
                existing_tracked_feature.add(tracked_feature_points[i])

                if existing_tracked_feature.is_full():

                    full_history_features_count += 1

                    if existing_tracked_feature.is_smooth() is False:

                        unsmooth_features_count += 1

            else:

                self.remove_tracked_feature(i)

        if unsmooth_features_count < (full_history_features_count / 2):

            # The majority of features is smooth. We downgrade unsmooth features
            self.apply_unsmooth_grades()


    def apply_unsmooth_grades(self):

        unsmooth_features_out_count = 0

        # for i, tracked_feature in enumerate(self._tracked_features):

        for i in reversed(xrange(len(self._tracked_features))):

            tracked_feature = self._tracked_features[i]

            tracked_feature.apply_score_change()

            if tracked_feature.is_out():
                self.remove_tracked_feature(i)
                unsmooth_features_out_count += 1

    def remove_tracked_feature(self, index):

        self._not_tracked_feature_count += 1

        # Remove the feature from the two lists at the specified index.
        self._tracked_features.pop(index)
        self._raw_tracked_feature_points.pop(index)

    def repopulate_feature_points(self, gray_image):

        new_raw_tracked_feature_points = self._optical_flow.find_features_to_track(gray_image)

        if len(new_raw_tracked_feature_points) == 0:
            print "ERROR: No feature points to track."
            return

        self._raw_tracked_feature_points.extend(new_raw_tracked_feature_points)

        for new_feature_point in new_raw_tracked_feature_points:

            tracked_feature = TrackedFeature()
            tracked_feature.add(new_feature_point)
            self._tracked_features.append(tracked_feature)

        self._initial_features_count = len(self._tracked_features)

        self._feature_repopulation_threshold = (self._initial_features_count * 9) / 10

        if self._feature_repopulation_threshold < 100:
            self._feature_repopulation_threshold = 100

        self._not_tracked_feature_count = 0

def draw_feature_locations_previous_current(tracked_features, image_canvas):

    for tracked_feature in tracked_features:

        draw_current_feature(tracked_feature.get_value(0), tracked_feature.is_full(), image_canvas)

        if tracked_feature.get_full_count() > 1:

            draw_previous_feature(tracked_feature.get_value(-1), tracked_feature.is_full(), image_canvas)
            cv2.line(image_canvas, (tracked_feature.get_value(-1)[0][0], tracked_feature.get_value(-1)[0][1]), (tracked_feature.get_value(0)[0][0], tracked_feature.get_value(0)[0][1]), (255, 0, 0), 2)


def draw_previous_feature(previous_feature_location, has_full_history, image_canvas):

    if has_full_history:

        # red
        cv2.circle(image_canvas, (previous_feature_location[0][0], previous_feature_location[0][1]), 5, (0, 0, 255), -1)

    else:

        # yellow
        cv2.circle(image_canvas, (previous_feature_location[0][0], previous_feature_location[0][1]), 5, (0, 255, 255), -1)

def draw_current_feature(current_feature_location, has_full_history, image_canvas):

    if has_full_history:

        # green
        cv2.circle(image_canvas, (current_feature_location[0][0], current_feature_location[0][1]), 5, (0, 255, 0), -1)

    else:
        # orange
        cv2.circle(image_canvas, (current_feature_location[0][0], current_feature_location[0][1]), 5, (0, 128, 255), -1)

def main():

    images = ["../eval_data/motion_images/wiltshire_outside_10cm/IMG1.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG2.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG3.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG4.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG5.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG6.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG7.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG8.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG9.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG10.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG11.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG12.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG13.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG14.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG15.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG16.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG17.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG18.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG19.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG20.JPG"]

    images2 = ["../eval_data/motion_images/flat_10cm/IMG1.JPG",
              "../eval_data/motion_images/flat_10cm/IMG2.JPG",
              "../eval_data/motion_images/flat_10cm/IMG3.JPG",
              "../eval_data/motion_images/flat_10cm/IMG4.JPG",
              "../eval_data/motion_images/flat_10cm/IMG5.JPG",
              "../eval_data/motion_images/flat_10cm/IMG6.JPG",
              "../eval_data/motion_images/flat_10cm/IMG7.JPG",
              "../eval_data/motion_images/flat_10cm/IMG8.JPG",
              "../eval_data/motion_images/flat_10cm/IMG9.JPG",
              "../eval_data/motion_images/flat_10cm/IMG10.JPG",
              "../eval_data/motion_images/flat_10cm/IMG11.JPG",
              "../eval_data/motion_images/flat_10cm/IMG12.JPG"]


    vo = VisualOdometer()

    for image_filepath in images:

        new_image = cv2.imread(image_filepath)

        vo.process_image(image_filepath)

        draw_feature_locations_previous_current(vo._tracked_features, new_image)

        cv2.imshow("result", new_image)

        cv2.waitKey(2000)

    cv2.waitKey()

    # buffer = CircularBuffer(7)
    #
    # print buffer._history
    #
    # buffer.add(1)
    #
    # print buffer._history
    #
    # buffer.add(2)
    #
    # print buffer._history
    #
    # buffer.add(3)
    #
    # print buffer._history
    #
    # buffer.add(4)
    # buffer.add(5)
    # buffer.add(6)
    # buffer.add(7)
    # buffer.add(8)
    #
    #
    # # buffer.add(9)
    #
    # # print buffer.get_value(buffer._current_index)
    #
    # # print buffer._history[buffer._current_index]
    # # print buffer._history[buffer._current_index - 1]
    #
    # buffer.add(9)
    #
    # print buffer._history[buffer._current_index]buffer.add(9)
    #
    # print buffer._history[buffer._current_index]
    # print buffer._history[buffer._current_index - 1]
    #
    # print buffer.get_value_current(0)
    # print buffer.get_value_current(-1)
    # print buffer._history[buffer._current_index - 1]
    #
    # print buffer.get_value_current(0)
    # print buffer.get_value_current(-1)
    #
    # # print buffer.get_value(-1)
    # # print buffer.get_value(-2)
    #
    # # print buffer._history[buffer.get_next_index() - 1]
    # # print buffer._history[buffer.get_next_index() - 2]
    # # print buffer._history[buffer.get_next_index() - 3]
    #
    # print buffer._history



if __name__ == "__main__":
    main()


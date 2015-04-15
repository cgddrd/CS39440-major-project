__author__ = 'connorgoddard'

import math
import numpy as np

from history_buffer import HistoryBuffer


class TrackedFeature(HistoryBuffer):
    _history_threshold = 7
    _is_smooth = False
    _smooth_transition_limit = 0.25
    _smooth_rotation_limit = 30.0 * math.pi / 180
    _smooth_score = 0
    _smooth_score_change = 0

    def __init__(self):
        HistoryBuffer.__init__(self, self._history_threshold)

    def add(self, new_value):

        super(TrackedFeature, self).add(new_value)

        if self.is_full():
            self.grade_smoothness()

    def grade_smoothness(self):

        current_point = self.get_value(0)[0]
        prev_point = self.get_value(-1)[0]
        second_point = self.get_value(-2)[0]
        sixth_point = self.get_value(-6)[0]

        # One back in history
        dx1 = current_point[0] - prev_point[0]
        dy1 = current_point[1] - prev_point[1]
        direction1 = math.atan2(dy1, dx1)

        # Two back in history
        dx2 = current_point[0] - second_point[0]
        dy2 = current_point[1] - second_point[1]
        direction2 = math.atan2(dy2, dx2)

        # Six back in history
        dx6 = current_point[0] - sixth_point[0]
        dy6 = current_point[1] - sixth_point[1]
        direction6 = math.atan2(dy6, dx6)

        # simple distance (approximation of real Euclidean distance)
        distance2 = abs(dx2) + abs(dy2)
        distance1 = abs(dx1) + abs(dy1)

        direction_change_6_to_2 = abs(self.subtract_angles(direction6, direction2))
        direction_change_2_to_1 = abs(self.subtract_angles(direction2, direction1))

        prev_change_is_smooth = (distance2 < self._smooth_transition_limit) or (
        direction_change_6_to_2 < self._smooth_rotation_limit)
        current_change_is_smooth = (distance1 < self._smooth_transition_limit) or (
        direction_change_2_to_1 < self._smooth_rotation_limit)

        # dxtest = current_point[0] - current_point[0]
        # dytest = current_point[1] - prev_point[1]

        # directiontest = TrackedFeature.angle_between((dx1, dx2), (dxtest, dytest))

        if (prev_change_is_smooth is True) and (current_change_is_smooth is True):

            self._is_smooth = True
            self._smooth_score_change = -1

        else:

            self._is_smooth = False

            if (current_change_is_smooth is True) and (prev_change_is_smooth is False):

                self._smooth_score_change = 5

            else:

                self._smooth_score_change = 8

        # if math.degrees(directiontest) > 30:
        #     print "oh dear"
        #     self._smooth_score_change = 100


    @staticmethod
    def subtract_angles(angle_1, angle_2):

        delta = angle_1 - angle_2

        while delta >= math.pi:
            delta -= math.pi

        while delta < -math.pi:
            delta += math.pi

        return delta

    @staticmethod
    def unit_vector(vector):

        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between(v1, v2):

        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = TrackedFeature.unit_vector(v1)
        v2_u = TrackedFeature.unit_vector(v2)
        angle = np.arccos(np.dot(v1_u, v2_u))
        if np.isnan(angle):
            if (v1_u == v2_u).all():
                return 0.0
            else:
                return np.pi
        return angle

    def apply_score_change(self):
        self._smooth_score += self._smooth_score_change

    def is_smooth(self):
        return self._is_smooth

    def is_out(self):
        return self._smooth_score > 10

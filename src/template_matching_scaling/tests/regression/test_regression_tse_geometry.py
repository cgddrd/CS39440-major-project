from unittest import TestCase
from nose.tools import *
from tse.tse_geometry import TSEGeometry
import numpy as np

__author__ = 'connorgoddard'


class TestRegressionTSEGeometry(TestCase):

    def test_calc_line_points_reflection(self):

        image_height = 200
        image_width = 300

        image_x_centre = image_width / 2

        origin_startpoint = (75, 0)
        origin_endpoint = (0, 5)

        expected_origin_line = [(75, 0), (60, 1), (45, 2), (30, 3), (15, 4), (0, 5)]
        expected_origin_line_reflected = [(225, 0), (240, 1), (255, 2), (270, 3), (285, 4), (300, 5)]

        calculated_lines = TSEGeometry.calc_line_points_horizontal_reflection(origin_startpoint, origin_endpoint, image_x_centre, origin_endpoint[1])

        assert_true(np.array_equal(calculated_lines[0], expected_origin_line))
        assert_true(np.array_equal(calculated_lines[1], expected_origin_line_reflected))

    def test_calc_line_points(self):

        startpoint_1 = (75, 0)
        endpoint_1 = (0, 5)

        startpoint_2 = (225, 0)
        endpoint_2 = (300, 5)

        expected_origin_line_1 = [(75, 0), (60, 1), (45, 2), (30, 3), (15, 4), (0, 5)]
        expected_origin_line_2 = [(225, 0), (240, 1), (255, 2), (270, 3), (285, 4), (300, 5)]

        calculated_lines = TSEGeometry.calc_line_points(startpoint_1, endpoint_1, startpoint_2, endpoint_2, endpoint_2[1])

        assert_true(np.array_equal(calculated_lines[0], expected_origin_line_1))
        assert_true(np.array_equal(calculated_lines[1], expected_origin_line_2))
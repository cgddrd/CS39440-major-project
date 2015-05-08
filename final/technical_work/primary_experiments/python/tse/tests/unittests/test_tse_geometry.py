from unittest import TestCase
from nose.tools import *
from tse.tse_geometry import TSEGeometry
import numpy as np
import math

__author__ = 'connorgoddard'


class TestTSEGeometry(TestCase):

    def test_constructor(self):
        test_geometry = TSEGeometry()
        assert_true(test_geometry is not None)

    def test_calc_measure_scale_factor(self):

        current_value = 10
        target_value = 100

        current_value2 = 100
        target_value2 = 9

        assert_equal(TSEGeometry.calc_measure_scale_factor(current_value, target_value), 10.0)

        assert_equal(TSEGeometry.calc_measure_scale_factor(current_value2, target_value2), 0.09)

    def test_scale_coordinate_relative_centre(self):

        origin_point = (0, 0)
        centre_point = (10, 10)

        scale_factor = 20

        scaled_x = centre_point[0] + ((origin_point[0] - centre_point[0]) * scale_factor)
        scaled_y = centre_point[1] + ((origin_point[1] - centre_point[1]) * scale_factor)

        expected_result = (scaled_x, scaled_y)

        assert_equal(TSEGeometry.scale_coordinate_relative_centre(origin_point, centre_point, scale_factor), expected_result)

    def test_calc_vec_magnitude(self):

        point_1 = (10, 20)
        point_2 = (50, 5)

        calculated_vector = ((point_2[0] - point_1[0]), (point_2[1] - point_1[1]))

        expected_result = math.sqrt((calculated_vector[0] ** 2) + (calculated_vector[1] ** 2))

        assert_equal(TSEGeometry.calc_vec_magnitude(point_1, point_2), expected_result)

    def test_calc_line_points_reflection(self):

        image_width = 300
        image_x_centre = image_width / 2

        origin_startpoint = (75, 0)
        origin_endpoint = (0, 5)

        expected_origin_line = [(75, 0), (60, 1), (45, 2), (30, 3), (15, 4), (0, 5)]
        expected_origin_line_reflected = [(225, 0), (240, 1), (255, 2), (270, 3), (285, 4), (300, 5)]

        calculated_lines = TSEGeometry.calc_line_points_horizontal_reflection(origin_startpoint, origin_endpoint, image_x_centre, origin_endpoint[1])

        assert_true(np.array_equal(calculated_lines[0], expected_origin_line))
        assert_true(np.array_equal(calculated_lines[1], expected_origin_line_reflected))

    def test_calc_line_points_reflection_straight_line(self):

        image_width = 300
        image_x_centre = image_width / 2

        origin_startpoint = (75, 0)
        origin_endpoint = (75, 5)

        expected_origin_straight_line = [(75, 0), (75, 1), (75, 2), (75, 3), (75, 4), (75, 5)]
        expected_origin_straight_line_reflected = [(225, 0), (225, 1), (225, 2), (225, 3), (225, 4), (225, 5)]

        calculated_straight_lines = TSEGeometry.calc_line_points_horizontal_reflection(origin_startpoint, origin_endpoint, image_x_centre, origin_endpoint[1])

        assert_true(np.array_equal(calculated_straight_lines[0], expected_origin_straight_line))
        assert_true(np.array_equal(calculated_straight_lines[1], expected_origin_straight_line_reflected))

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

    def test_calc_line_points_straight_line(self):

        startpoint_1 = (75, 0)
        endpoint_1 = (75, 5)

        startpoint_2 = (225, 0)
        endpoint_2 = (225, 5)

        expected_origin_straight_line_1 = [(75, 0), (75, 1), (75, 2), (75, 3), (75, 4), (75, 5)]
        expected_origin_straight_line_2 = [(225, 0), (225, 1), (225, 2), (225, 3), (225, 4), (225, 5)]

        calculated_straight_lines = TSEGeometry.calc_line_points(startpoint_1, endpoint_1, startpoint_2, endpoint_2, endpoint_2[1])

        assert_true(np.array_equal(calculated_straight_lines[0], expected_origin_straight_line_1))
        assert_true(np.array_equal(calculated_straight_lines[1], expected_origin_straight_line_2))
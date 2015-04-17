from unittest import TestCase
from nose.tools import *
from tse.tse_datautils import TSEDataUtils
import numpy as np


__author__ = 'connorgoddard'


class TestTSEDataUtils(TestCase):

    def test_constructor(self):
        test_utils = TSEDataUtils()
        assert_true(test_utils is not None)

    def test_get_smallest_key_dict(self):
        test_dict = {-1: 'value_-1', 1: 'value_1', 2: 'value_2'}
        assert_equal(TSEDataUtils.get_smallest_key_dict(test_dict), -1, "Smallest key should equal 2")

    def test_get_largest_key_dict(self):
        test_dict = {-1: 'value_-1', 1: 'value_1', 2: 'value_2'}
        assert_equal(TSEDataUtils.get_largest_key_dict(test_dict), 2, "Largest key should equal 2")

    def test_get_smallest_key_value_dict(self):
        test_dict = {-1: 'value_-1', 1: 'value_1', 2: 'value_2'}
        assert_equal(TSEDataUtils.get_smallest_key_value_dict(test_dict), "value_-1",
                     "Smallest value should equal 'value_-1'")

    def test_string_list_to_int_list(self):
        test_string_2d_list = [["1", "2"], ["3", "4"]]

        test_string_2d_to_int_2d_list = TSEDataUtils.string_2d_list_to_int_2d_list(test_string_2d_list)

        assert_equal(test_string_2d_list[0][0] + test_string_2d_list[0][1], "12")
        assert_equal(test_string_2d_to_int_2d_list[0][0] + test_string_2d_to_int_2d_list[0][1], 3)

        assert_equal(test_string_2d_list[1][0] + test_string_2d_list[1][1], "34")
        assert_equal(test_string_2d_to_int_2d_list[1][0] + test_string_2d_to_int_2d_list[1][1], 7)

    def test_convert_to_int(self):
        test_string_list = ["1", "2", "3", "4"]

        test_string_to_int_list = TSEDataUtils.convert_list_to_int(test_string_list)

        assert_equal(test_string_list[0] + test_string_list[1], "12")
        assert_equal(test_string_list[2] + test_string_list[3], "34")
        assert_equal(test_string_to_int_list[0] + test_string_to_int_list[1], 3)
        assert_equal(test_string_to_int_list[2] + test_string_to_int_list[3], 7)

    def test_calc_moving_average_array(self):
        test_data = [1, 2, 3, 4, 5, 6]

        points_to_average = 2

        expected_result = np.array([float((2+1) / 2.0), float((3+2)/2.0), float((4+3)/2.0), float((5+4)/2.0), float((6+5)/2.0)])

        # Using both Numpy test asserts and nosetest asserts.
        np.testing.assert_equal(TSEDataUtils.calc_moving_average(test_data, points_to_average), expected_result)
        assert_true(np.array_equal(TSEDataUtils.calc_moving_average(test_data, points_to_average), expected_result))

    def test_convert_array_to_numpy_array(self):
        test_data = [1, 2, 3, 4, 5, 6]

        expected_result = np.array(test_data)

        assert_true(np.array_equal(TSEDataUtils.convert_array_to_numpy_array(test_data), expected_result))

    def test_calc_cartesian_product(self):
        test_data1 = [1, 2]
        test_data2 = [3, 4, 5]

        test_data3 = [10, 20, 30]
        test_data4 = [40, 50]

        expected_result = [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
        expected_result2 = [[10, 40], [10, 50], [20, 40], [20, 50], [30, 40], [30, 50]]

        assert_true(np.array_equal(TSEDataUtils.calc_cartesian_product([test_data1, test_data2]), expected_result))
        assert_true(np.array_equal(TSEDataUtils.calc_cartesian_product([test_data3, test_data4]), expected_result2))
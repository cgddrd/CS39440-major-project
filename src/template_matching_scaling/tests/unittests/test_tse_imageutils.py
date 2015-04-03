from unittest import TestCase
from nose.tools import *
from tse.tse_imageutils import TSEImageUtils
from tse.tse_point import TSEPoint
import numpy as np
import math
import cv2

__author__ = 'connorgoddard'


class TestTSEImageUtils(TestCase):

    def test_constructor(self):
        test_imageutils = TSEImageUtils()
        assert_true(test_imageutils is not None)

    def test_calc_euclidean_distance_cv2_norm(self):
        image_1 = np.zeros((200, 200, 3), dtype=np.uint8)
        image_2 = image_1
        image_3 = np.full((200, 200, 3), [0, 0, 200],  dtype=np.uint8)

        # Check that for perfectly matching images, we get a score of exactly 0.
        assert_equal(TSEImageUtils.calc_euclidean_distance_cv2_norm(image_1, image_2), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_euclidean_distance_cv2_norm(image_1, image_3) > 0)

    def test_calc_scaled_image_pixel_dimension_coordinates_rounded(self):

        dimension_max_val = 20

        scale_factor = 0.5

        expected_result_rounded = np.rint(np.arange(0, dimension_max_val) * scale_factor)

        assert_true(np.array_equal(expected_result_rounded, TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(dimension_max_val, scale_factor, round=True)))

    def test_calc_scaled_image_pixel_dimension_coordinates(self):

        dimension_max_val = 20

        scale_factor = 0.5

        expected_result_float = np.arange(0, dimension_max_val) * scale_factor

        assert_true(np.array_equal(expected_result_float, TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(dimension_max_val, scale_factor, round=False)))

    def test_reshape_match_images(self):

        image_target_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_current_shape = np.zeros(120000, dtype=np.uint8)

        assert_false(np.array_equal(image_target_shape, image_current_shape))

        image_reshaped = TSEImageUtils.reshape_match_images(image_current_shape, image_target_shape)

        assert_true(np.array_equal(image_target_shape, image_reshaped))

    def test_reshape_match_images_same_shape(self):

        image_target_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_current_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_reshaped = TSEImageUtils.reshape_match_images(image_current_shape, image_target_shape)

        assert_true(np.array_equal(image_target_shape, image_reshaped))


    def test_extract_rows_cols_pixels_image(self):

        required_rows = [1, 100]
        required_cols = [10, 20]

        image_target_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_target_shape[1, 10] = [0, 0, 200]
        image_target_shape[1, 20] = [0, 0, 200]

        image_target_shape[100, 10] = [0, 0, 200]
        image_target_shape[100, 20] = [0, 0, 200]

        returned_image = TSEImageUtils.extract_rows_cols_pixels_image(required_rows, required_cols, image_target_shape)

        # '.all()' loops through every element in 'returned_image' and checks that they equal '[0, 0, 200]'
        assert_true((returned_image == [0, 0, 200]).all())

    def test_convert_hsv_and_remove_luminance(self):

        image_three_channel = np.full((50, 50, 3), [100, 50, 200], dtype=np.uint8)

        image_two_channel = TSEImageUtils.convert_hsv_and_remove_luminance(image_three_channel)

        # Test that the colour space has been converted to HSV, and the 'V' channel has been set to 0 (i.e. remove it)
        assert_true((image_two_channel == [170, 191, 0]).all(), "Converted colour to HSV and 'V' stripped should equal [170, 191, 0]")

    def test_scale_image_roi_relative_centre(self):

        origin_point = (0, 0)
        end_point = (10, 10)

        centre_point = (5, 5)

        scale_factor = 20

        scaled_origin_x = centre_point[0] + ((origin_point[0] - centre_point[0]) * scale_factor)
        scaled_origin_y = centre_point[1] + ((origin_point[1] - centre_point[1]) * scale_factor)

        scaled_end_x = centre_point[0] + ((end_point[0] - centre_point[0]) * scale_factor)
        scaled_end_y = centre_point[1] + ((end_point[1] - centre_point[1]) * scale_factor)

        result = TSEImageUtils.scale_image_roi_relative_centre(origin_point, end_point, scale_factor)

        assert_equal((scaled_origin_x, scaled_origin_y), result[0].to_tuple())
        assert_equal((scaled_end_x, scaled_end_y), result[1].to_tuple())

    def test_extract_image_sub_window(self):

        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        # Set 50x50px section to Red (top-left corner)
        test_image[0:50, 0:50] = [0, 0, 200]

        expected_result = np.full((50, 50, 3), [0, 0, 200], dtype=np.uint8)

        extracted_slice = TSEImageUtils.extract_image_sub_window(test_image, TSEPoint(0, 0), TSEPoint(50, 50))

        assert_true(np.array_equal(expected_result, extracted_slice))
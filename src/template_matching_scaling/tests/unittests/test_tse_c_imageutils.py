from unittest import TestCase
from nose.tools import *

from tse_compiled.tse_c_imageutils import TSECImageUtils
from tse.tse_geometry import TSEGeometry
from tse.tse_utils import TSEUtils

import numpy as np

__author__ = 'connorgoddard'


class TestTSECImageUtils(TestCase):
    def test_calc_ssd_slow(self):
        # Create a sample test image that is empty.
        original_image = np.full((400, 400, 3), [0, 200, 0],  dtype=np.uint8)

        # Calculate the scale factor (MAKING SURE TO SUBTRACT '1' from the max height/width to account for array index out of bounds issue)
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(200, (400 - 1))

        # Calculate the scaled indices to identify the pixels in the larger image that we will want to make GREEN to provide evidence for the test succeeding.
        original_image_scaled_indices = np.rint((np.arange(0, 200) * scale_factor_width)).astype(int)

        rows_cols_cartesian_product = np.hsplit(TSEUtils.calc_cartesian_product([original_image_scaled_indices, original_image_scaled_indices]), 2)

        rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        # We now want to set each fo the pixels THAT WE EXPECT TO BE EXTRACTED BY THE TEST to GREEN to show that the test has passed.
        original_image[rows_to_extract, cols_to_extract] = [0, 200, 0]

        # Once we have performed the pixel extraction, we expect that all of the pixels returned will be GREEN (based ont he setup above)
        matching_image = np.full((200, 200, 3), [0, 200, 0],  dtype=np.uint8)

        non_matching_image = np.full((200, 200, 3), [200, 0, 0],  dtype=np.uint8)

        # Check that for perfectly matching images, we get a score of exactly 0.
        assert_equal(TSECImageUtils.calc_ssd_slow(matching_image, original_image, matching_image.shape[0], matching_image.shape[1], scale_factor_width, scale_factor_width), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSECImageUtils.calc_ssd_slow(non_matching_image, original_image, matching_image.shape[0], matching_image.shape[1], scale_factor_width, scale_factor_width) > 0)

    def test_extract_rows_cols_pixels_image(self):
        required_rows = np.array([1, 100]).astype(float)
        required_cols = np.array([10, 20]).astype(float)

        image_target_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_target_shape[1, 10] = [0, 0, 200]
        image_target_shape[1, 20] = [0, 0, 200]

        image_target_shape[100, 10] = [0, 0, 200]
        image_target_shape[100, 20] = [0, 0, 200]

        returned_image = TSECImageUtils.extract_rows_cols_pixels_image(required_rows, required_cols, image_target_shape)

        # '.all()' loops through every element in 'returned_image' and checks that they equal '[0, 0, 200]'
        assert_true((returned_image == [0, 0, 200]).all())

    def test_calc_cartesian_product(self):

        # NOTE: As we are creating these arrays MANUALLY for the test, we have to cast them to NUMPY ARRAYS first.
        test_data1 = np.array([1, 2])
        test_data2 = np.array([3, 4, 5])

        test_data3 = np.array([10, 20, 30])
        test_data4 = np.array([40, 50])

        expected_result = [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
        expected_result2 = [[10, 40], [10, 50], [20, 40], [20, 50], [30, 40], [30, 50]]

        assert_true(np.array_equal(TSECImageUtils.calc_cartesian_product([test_data1, test_data2]), expected_result))
        assert_true(np.array_equal(TSECImageUtils.calc_cartesian_product([test_data3, test_data4]), expected_result2))

    def test_reshape_match_images(self):

        image_target_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_current_shape = np.zeros((120000, 1, 1), dtype=np.uint8)

        assert_false(np.array_equal(image_target_shape, image_current_shape))

        image_reshaped = TSECImageUtils.reshape_match_images(image_current_shape, image_target_shape)

        assert_true(np.array_equal(image_target_shape, image_reshaped))

    def test_reshape_match_images_same_shape(self):

        image_target_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_current_shape = np.zeros((200, 200, 3), dtype=np.uint8)

        image_reshaped = TSECImageUtils.reshape_match_images(image_current_shape, image_target_shape)

        assert_true(np.array_equal(image_target_shape, image_reshaped))
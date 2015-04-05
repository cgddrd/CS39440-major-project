from unittest import TestCase
from nose.tools import *
from tse.tse_imageutils import TSEImageUtils
from tse.tse_geometry import TSEGeometry
from tse.tse_utils import TSEUtils
import numpy as np

__author__ = 'connorgoddard'


class TestRegressionTSEImageUtils(TestCase):

    def setUp(self):

        # Create a sample test image that is empty.
        self._original_image = np.zeros((400, 400, 3), dtype=np.uint8)

        # Calculate the scale factor (MAKING SURE TO SUBTRACT '1' from the max height/width to account for array index out of bounds issue)
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(200, (400 - 1))

        # Calculate the scaled indices to identify the pixels in the larger image that we will want to make GREEN to provide evidence for the test succeeding.
        original_image_scaled_indices = np.rint((np.arange(0, 200) * scale_factor_width)).astype(int)

        rows_cols_cartesian_product = np.hsplit(TSEUtils.calc_cartesian_product([original_image_scaled_indices, original_image_scaled_indices]), 2)

        rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        # We now want to set each fo the pixels THAT WE EXPECT TO BE EXTRACTED BY THE TEST to GREEN to show that the test has passed.
        self._original_image[rows_to_extract, cols_to_extract] = [0, 200, 0]


    # Refs: Fix #94 - https://github.com/cgddrd/CS39440-major-project/issues/94
    def test_calc_ed_template_match_score_scaled_fix_94(self):

        # Once we have performed the pixel extraction, we expect that all of the pixels returned will be GREEN (based ont he setup above)
        matching_image = np.full((200, 200, 3), [0, 200, 0],  dtype=np.uint8)

        non_matching_image = np.full((200, 200, 3), [200, 0, 0],  dtype=np.uint8)

        # Check that for perfectly matching images, we get a score of exactly 0.
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled(matching_image, self._original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled(non_matching_image, self._original_image) > 0)
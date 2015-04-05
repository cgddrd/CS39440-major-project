from unittest import TestCase
from nose.tools import *
from tse.tse_imageutils import TSEImageUtils
from tse.tse_point import TSEPoint
from tse.tse_geometry import TSEGeometry
from tse.tse_utils import TSEUtils
import numpy as np
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

    def test_calc_template_match_compare_cv2_score_SQDIFF(self):
        image_1 = np.zeros((200, 200, 3), dtype=np.uint8)
        image_2 = image_1
        image_3 = np.full((200, 200, 3), [0, 0, 200],  dtype=np.uint8)

        # Check that for perfectly matching images, we get a score of exactly 0.
        assert_equal(TSEImageUtils.calc_template_match_compare_cv2_score(image_1, image_2, cv2.cv.CV_TM_SQDIFF), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_template_match_compare_cv2_score(image_1, image_3, cv2.cv.CV_TM_SQDIFF) > 0)

    def test_calc_compare_histogram_CHISQR(self):
        image_1 = np.zeros((200, 200, 3), dtype=np.uint8)
        image_2 = image_1
        image_3 = np.full((200, 200, 3), [100, 0, 0],  dtype=np.uint8)

        # Check that for perfectly matching images, we get a score of exactly 0.
        assert_equal(TSEImageUtils.calc_compare_hsv_histogram(image_1, image_2, cv2.cv.CV_COMP_CHISQR), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_compare_hsv_histogram(image_1, image_3, cv2.cv.CV_COMP_CHISQR) > 0)

    def test_calc_compare_histogram_CORREL(self):
        image_1 = np.zeros((200, 200, 3), dtype=np.uint8)
        image_2 = image_1
        image_3 = np.full((200, 200, 3), [100, 0, 0],  dtype=np.uint8)

        matching_result = TSEImageUtils.calc_compare_hsv_histogram(image_1, image_2, cv2.cv.CV_COMP_CORREL)
        non_matching_result = TSEImageUtils.calc_compare_hsv_histogram(image_1, image_3, cv2.cv.CV_COMP_CORREL)

        # Matching result should be greater than a non-matching result for CORREL matching method.
        assert_true(matching_result > non_matching_result)

    def test_calc_template_match_compare_cv2_score_CCORR(self):
        image_1 = np.full((200, 200, 3), [0, 200, 0],  dtype=np.uint8)
        image_2 = image_1
        image_3 = np.full((200, 200, 3), [200, 0, 0],  dtype=np.uint8)

        matching_result = TSEImageUtils.calc_template_match_compare_cv2_score(image_1, image_2, cv2.cv.CV_TM_CCORR_NORMED)
        non_matching_result = TSEImageUtils.calc_template_match_compare_cv2_score(image_1, image_3, cv2.cv.CV_TM_CCORR_NORMED)

        # For CCORR, we would expect that a perfectly matching image will score HIGHER than a non-matching image
        assert_true(matching_result > non_matching_result)

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

    def test_calc_ed_template_match_score_scaled(self):

        # Create a sample test image that is empty.
        original_image = np.zeros((400, 400, 3), dtype=np.uint8)

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
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled(matching_image, original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled(non_matching_image, original_image) > 0)

    def test_calc_ed_template_match_score_scaled_slow(self):

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
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled_slow(matching_image, original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled_slow(non_matching_image, original_image) > 0)

    def test_calc_ed_template_match_score_scaled_compiled(self):

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
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled_compiled(matching_image, original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled_compiled(non_matching_image, original_image) > 0)

    def test_calc_ed_template_match_score_scaled_compiled_slow(self):

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
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled_compiled_slow(matching_image, original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled_compiled_slow(non_matching_image, original_image) > 0)

    def test_calc_ed_template_match_score_scaled_compiled(self):

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
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled_compiled(matching_image, original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled_compiled(non_matching_image, original_image) > 0)

    def test_calc_ed_template_match_score_scaled_compiled_slow(self):

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
        assert_equal(TSEImageUtils.calc_ed_template_match_score_scaled_compiled_slow(matching_image, original_image), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_ed_template_match_score_scaled_compiled_slow(non_matching_image, original_image) > 0)

    def test_calc_template_match_compare_cv2_score_scaled_SQDIFF(self):

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
        assert_equal(TSEImageUtils.calc_template_match_compare_cv2_score_scaled(matching_image, original_image, cv2.cv.CV_TM_SQDIFF), 0)

        # Check that for non-matching images, we get a score > 0.
        assert_true(TSEImageUtils.calc_template_match_compare_cv2_score_scaled(non_matching_image, original_image, cv2.cv.CV_TM_SQDIFF) > 0)

    def test_calc_template_match_compare_cv2_score_scaled_CCORR_NORMED(self):

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

        # Check that for perfectly matching images, we get a score of exactly 1.0 (normalised - higher score = better match).
        assert_equal(TSEImageUtils.calc_template_match_compare_cv2_score_scaled(matching_image, original_image, cv2.cv.CV_TM_CCORR_NORMED), 1.0)

        # Check that for non-matching images, we get a score < 1.0 (should get a smaller score for non-matchign images).
        assert_true(TSEImageUtils.calc_template_match_compare_cv2_score_scaled(non_matching_image, original_image, cv2.cv.CV_TM_CCORR_NORMED) < 1.0)

    def test_scale_image_interpolation_auto(self):

        original_image = np.full((200, 200, 3), [0, 200, 0],  dtype=np.uint8)

        # Set the centre pixel fo the original image to a different colour for comparison once scaling is complete.
        original_image[100, 100] = [200, 0, 0]

        larger_target_image = np.zeros((400, 400, 3),  dtype=np.uint8)

        scaled_result = TSEImageUtils.scale_image_interpolation_auto(original_image, larger_target_image)

        # We would expect the centre pixel of the scaled image NOT to be GREEN, as in the original non-scaled image this was set to RED.
        assert_false(np.array_equal(scaled_result[200, 200], [0, 200, 0]))

        # We would expect all other pixels (apart from immediate neighbours around the centre pixel due to the interpolation) to still be GREEN.
        assert_true(np.array_equal(scaled_result[0, 0], [0, 200, 0]))
        assert_true(np.array_equal(scaled_result[399, 399], [0, 200, 0]))
        assert_true(np.array_equal(scaled_result[195, 195], [0, 200, 0]))
        assert_true(np.array_equal(scaled_result[205, 205], [0, 200, 0]))

    def test_scale_hsv_image_no_interpolation_auto(self):

        original_image = np.full((200, 200, 3), [0, 200, 0],  dtype=np.uint8)

        # Set the centre pixel fo the original image to a different colour for comparison once scaling is complete.
        original_image[100, 100] = [200, 0, 0]

        larger_target_image = np.zeros((400, 400, 3),  dtype=np.uint8)

        scaled_result = TSEImageUtils.scale_image_no_interpolation_auto(original_image, larger_target_image)

        # We would expect the centre pixel of the scaled image to be RED, as in the original non-scaled image this was set to RED.
        assert_true(np.array_equal(scaled_result[200, 200], [200, 0, 0]))

        # We would expect all other pixels to still be GREEN.
        assert_true(np.array_equal(scaled_result[0, 0], [0, 200, 0]))

    def test_scale_image_interpolation_man(self):

        original_image = np.full((200, 200, 3), [0, 200, 0],  dtype=np.uint8)

        # Set the centre pixel fo the original image to a different colour for comparison once scaling is complete.
        original_image[100, 100] = [200, 0, 0]

        larger_target_image = np.zeros((400, 400, 3),  dtype=np.uint8)

        # Calculate the scale factor based on the widths of the two images (as the width/height are equal, we can just use the width)
        scale_factor = TSEGeometry.calc_measure_scale_factor(original_image.shape[1], (larger_target_image.shape[1]))

        scaled_result = TSEImageUtils.scale_image_interpolation_man(original_image, scale_factor)

        # We would expect the centre pixel of the scaled image NOT to be GREEN, as in the original non-scaled image this was set to RED.
        assert_false(np.array_equal(scaled_result[200, 200], [0, 200, 0]))

        # We would expect all other pixels (apart from immediate neighbours around the centre pixel due to the interpolation) to still be GREEN.
        assert_true(np.array_equal(scaled_result[0, 0], [0, 200, 0]))
        assert_true(np.array_equal(scaled_result[399, 399], [0, 200, 0]))
        assert_true(np.array_equal(scaled_result[195, 195], [0, 200, 0]))
        assert_true(np.array_equal(scaled_result[205, 205], [0, 200, 0]))

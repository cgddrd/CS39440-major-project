"""

Module Name: TSEImageUtils

Description: Provides functionality relating to image processing and template matching.

"""
from __future__ import division

import math

import cv2
import numpy as np

from tse_compiled.tse_c_imageutils import TSECImageUtils
from tse.tse_geometry import TSEGeometry
from tse_datautils import TSEDataUtils
from tse_point import TSEPoint


__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


class TSEImageUtils:
    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_euclidean_distance_cv2_norm(image_1, image_2):
        """
        Calculates the Euclidean Distance similarity between two images via the OpenCV 'norm' function.

        For Euclidean Distance, the 'cv2.NORM_L2' parameter is passed into 'norm'.

        :param image_1: The first image to compare.
        :param image_2: The second image to compare.
        :return Float containing the calculated Euclidean Distance score between the two images.
        """

        return cv2.norm(image_1, image_2, cv2.NORM_L2)

    @staticmethod
    def calc_ed_template_match_score_scaled(template_patch, scaled_search_window):
        """
        Calculates the Euclidean Distance similarity between two images, geometrically-scaling the first image as
        appropriate in order to match the dimensions of the second image.

        :param template_patch: The smaller template image that we are looking for.
        :param scaled_search_window: The larger search window image that we need to match in terms of scaling the template patch.
        :return Float containing the calculated Euclidean Distance score between the scaled images.
        """

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        """
        We have to subtract '-1' from the width and height of the scaled search window BEFORE calculating the scale factor to multiply by.

        This is to ensure that the scaled pixel coordinates remain INSIDE THE BOUNDS of the array indices. (i.e. for height = 200, max index = 199)

        e.g.
        ---

        - Search Window Image = (200x200px) (0-199 array index range)

        - Target Template Patch Row Index = 100

        - Scale Factor = 2.0

        - 100 * 2.0 = 200.00 (int: 200)

        - 200 > 199 max image index = ERROR

        """

        # Calculate the scale factor between both image heights and widths.
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, (scaled_search_window_width - 1))
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, (scaled_search_window_height - 1))

        # Calculate the scaled coordinates of each pixel in the template patch ready fo extraction form the larger search window.
        scaled_window_heights = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_height, scale_factor_height)
        scaled_window_widths = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_width, scale_factor_width)

        # Extract all the pixels from the larger search window corresponding to the scaled coordinates of each pixel in the template patch.
        search_window_target_pixels = TSEImageUtils.extract_rows_cols_pixels_image(scaled_window_heights, scaled_window_widths, scaled_search_window)

        # Reshape the Numpy array so that we match the structure of the original template patch (required to calculate ED)
        reshaped_search_window_target_pixels = TSEImageUtils.reshape_match_images(search_window_target_pixels, template_patch)

        # Calculate the Euclidean Distance between the template patch, and the new image representing the extracted pixels.
        return TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, reshaped_search_window_target_pixels)

    @staticmethod
    def calc_ed_template_match_score_scaled_slow(template_patch, scaled_search_window):
        """
        Alternative method to calculate the Euclidean Distance similarity between two images, geometrically-scaling the first image as
        appropriate in order to match the dimensions of the second image.

        This function instead attempts to loop through each template patch pixel in turn, calculating the coordinate of the scaled
        pixel position, before calculating Sum of Squared Differences value between the two "on the fly". Following investigations, this approach performed significantly
        slower than the previous function.

        :param template_patch: The smaller template image that we are looking for.
        :param scaled_search_window: The larger search window image that we need to match in terms of scaling the template patch.
        :return Float containing the calculated Euclidean Distance score between the scaled images.
        """

        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        """
        We have to subtract '-1' from the width and height of the scaled search window BEFORE calculating the scale factor to multiply by.

        This is to ensure that the scaled pixel coordinates remain INSIDE THE BOUNDS of the array indices. (i.e. for height = 200, max index = 199)

        e.g.
        ---

        - Search Window Image = (200x200px) (0-199 array index range)

        - Target Template Patch Row Index = 100

        - Scale Factor = 2.0

        - 100 * 2.0 = 200.00 (int: 200)

        - 200 > 199 max image index = ERROR

        """
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, (scaled_search_window_width - 1))
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, (scaled_search_window_height - 1))

        ssd = 0

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(template_patch_height):
            for j in xrange(template_patch_width):

                template_patch_val_channel_1 = template_patch.item(i, j, 0)
                template_patch_val_channel_2 = template_patch.item(i, j, 1)
                template_patch_val_channel_3 = template_patch.item(i, j, 2)

                # Extract the colour data from the pixel in the corresponding scaled position within the larger search image.
                scaled_search_window_val_channel_1 = scaled_search_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 0)
                scaled_search_window_val_channel_2 = scaled_search_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 1)
                scaled_search_window_val_channel_3 = scaled_search_window.item((i * scale_factor_height),
                                                                           (j * scale_factor_width), 2)

                # Calculate Sum of Squared Differences similarity value between corresponding colour values.
                diff_channel_1 = template_patch_val_channel_1 - scaled_search_window_val_channel_1

                diff_channel_2 = template_patch_val_channel_2 - scaled_search_window_val_channel_2

                diff_channel_3 = template_patch_val_channel_3 - scaled_search_window_val_channel_3

                ssd += (diff_channel_1 * diff_channel_1)

                ssd += (diff_channel_2 * diff_channel_2)

                ssd += (diff_channel_3 * diff_channel_3)

        return math.sqrt(ssd)

    @staticmethod
    def calc_ed_template_match_score_scaled_compiled(template_patch, scaled_search_window):
        """
        Most efficient method to calculate the Euclidean Distance similarity between two images, geometrically-scaling the first image as
        appropriate in order to match the dimensions of the second image.

        This function delegates scaling and Euclidean Distance matching to the more efficient 'TSECImageUtils' module
        that is translated to compiled C code using the Cython library.

        :param template_patch: The smaller template image that we are looking for.
        :param scaled_search_window: The larger search window image that we need to match in terms of scaling the template patch.
        :return Float containing the calculated Euclidean Distance score between the scaled images.
        """
        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        """
        We have to subtract '-1' from the width and height of the scaled search window BEFORE calculating the scale factor to multiply by.

        This is to ensure that the scaled pixel coordinates remain INSIDE THE BOUNDS of the array indices. (i.e. for height = 200, max index = 199)

        e.g.
        ---

        - Search Window Image = (200x200px) (0-199 array index range)

        - Target Template Patch Row Index = 100

        - Scale Factor = 2.0

        - 100 * 2.0 = 200.00 (int: 200)

        - 200 > 199 max image index = ERROR

        """
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, (scaled_search_window_width - 1))
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, (scaled_search_window_height - 1))

        scaled_window_heights = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_height, scale_factor_height)
        scaled_window_widths = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_width, scale_factor_width)

        # Notice here that we are calling the 'TSECImageUtils' module - Cython C code compiled module.
        search_window_target_pixels = TSECImageUtils.extract_rows_cols_pixels_image(scaled_window_heights, scaled_window_widths, scaled_search_window)

        reshaped_search_window_target_pixels = TSECImageUtils.reshape_match_images(search_window_target_pixels, template_patch)

        return TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, reshaped_search_window_target_pixels), reshaped_search_window_target_pixels

    @staticmethod
    def calc_ed_template_match_score_scaled_compiled_slow(template_patch, scaled_search_window):
        """
        Slower method to calculate the Euclidean Distance similarity between two images, geometrically-scaling the first image as
        appropriate in order to match the dimensions of the second image. (Uses same method as 'TSEImageUtils.calc_ed_template_match_score_scaled_slow')

        This function delegates matching to the more efficient 'TSECImageUtils' module
        that is translated to compiled C code using the Cython library.

        :param template_patch: The smaller template image that we are looking for.
        :param scaled_search_window: The larger search window image that we need to match in terms of scaling the template patch.
        :return Float containing the calculated Euclidean Distance score between the scaled images.
        """
        template_patch_height, template_patch_width = template_patch.shape[:2]
        scaled_search_window_height, scaled_search_window_width = scaled_search_window.shape[:2]

        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, (scaled_search_window_width - 1))
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, (scaled_search_window_height - 1))

        ssd = TSECImageUtils.calc_ssd_slow(template_patch, scaled_search_window, template_patch_height, template_patch_width, scale_factor_height, scale_factor_width)

        return math.sqrt(ssd)

    @staticmethod
    def calc_scaled_image_pixel_dimension_coordinates(image_dim_end, scale_factor, image_dim_start=0, round=True):
        """
        Scales a range of pixel dimension coordinates (i.e. all X coordinates OR Y-coordinates) by a scaling factor.

        :param image_dim_end: The maximum pixel dimension value to scale (normally the last row or column in an image)
        :param scale_factor: The factor by which to scale by (Float)
        :param image_dim_start: The first pixel dimension value to scale (normally 0)
        :param round: Specifies whether scaled values should be rounded to the nearest int (which for image pixels should be true)
        :return Float containing the calculated Euclidean Distance score between the scaled images.
        """

        image_dim_coordinates = np.arange(image_dim_start, image_dim_end)

        # Perform Numpy element-wise multiplication function to all elements in array.
        scaled_image_dim_coordinates = image_dim_coordinates * scale_factor

        # As we are dealing in image pixels, we will normally want to round the results to the nearest integer.
        if round is True:
            return np.rint(scaled_image_dim_coordinates)

        return scaled_image_dim_coordinates

    @staticmethod
    def reshape_match_images(current_matrix, target_matrix):
        """
        If required, re-shapes the structure of two Numpy 'ndarray' objects representing images so that they match.

        :param current_matrix: The matrix object to be re-shaped.
        :param target_matrix: The matric object of whose shape is to be matched.
        :return The same Numpy 'ndarray' object, but with potentially a different internal dimension structure.
        """

        if current_matrix.shape != target_matrix.shape:

            # Re-shape one of the two matrices so that their internal dimension structures match.
            return current_matrix.reshape(target_matrix.shape)

        return current_matrix

    @staticmethod
    def extract_rows_cols_pixels_image(required_rows, required_cols, image):
        """
        Extracts the pixel defined at each specified row-column position within the image by calculating the cartesian product
        between the collection of specified rows and columns.

        :param required_rows: The collection of required image rows.
        :param required_rows: The collection of required image columns.
        :param image: The image from which the specified pixels are to be extracted.
        :return The subset of pixels from the original image as defined by the cartesian product
        between the collection of specified rows and columns.
        """

        # Get the cartesian product between the two then split into one array for all rows, and one array for all cols.
        rows_cols_cartesian_product = np.hsplit(TSEDataUtils.calc_cartesian_product([required_rows, required_cols]), 2)

        rows_to_extract = rows_cols_cartesian_product[0].astype(int)
        cols_to_extract = rows_cols_cartesian_product[1].astype(int)

        # Numpy library will extract ALL of the elements defined by the lists of rows and columns.
        return image[rows_to_extract, cols_to_extract]

    @staticmethod
    def calc_template_match_compare_cv2_score(image_1, image_2, match_method):
        """
        Performs appearance-based matching between two images utilising the OpenCV 'matchTemplate' function.

        :param image_1: The first image to match.
        :param image_2: The second image to match.
        :param match_method: Enum representing the similarity metric to be used ('CV_TM_CCORR_NORMED' for Normalised Cross-Correlation')
        :return Float representing the match score calculated by the similarity metric.
        """

        # Perform the actual template matching.
        res = cv2.matchTemplate(image_2, image_1, match_method)

        # '_' is a placeholder convention to indicate we do not want to use these returned values.
        # Obtain the lowest and highest match scores.
        min_val, max_val, _, _ = cv2.minMaxLoc(res)

        # If we are matching using SSD, then the lowest score is the "best match", so return this.
        if match_method == cv2.cv.CV_TM_SQDIFF or match_method == cv2.cv.CV_TM_SQDIFF_NORMED:
            return min_val

        # Otherwise (and in most cases), we will want to return the highest score.
        return max_val

    @staticmethod
    def calc_template_match_compare_cv2_score_scaled(template_image, current_search_window, match_method):
        """
        Performs geometric scaling of the template image in order to subsequently
        run appearance-based matching utilising the OpenCV 'matchTemplate' function.

        :param template_image: The template image whose pixels will be scaled.
        :param current_search_window: The larger search window.
        :param match_method: Enum representing the similarity metric to be used ('CV_TM_CCORR_NORMED' for Normalised Cross-Correlation')
        :return Float representing the match score calculated by the similarity metric.
        """

        template_patch_height, template_patch_width = template_image.shape[:2]
        scaled_search_window_height, scaled_search_window_width = current_search_window.shape[:2]

        """
        We have to subtract '-1' from the width and height of the scaled search window BEFORE calculating the scale factor to multiply by.

        This is to ensure that the scaled pixel coordinates remain INSIDE THE BOUNDS of the array indices. (i.e. for height = 200, max index = 199)

        e.g.
        ---

        - Search Window Image = (200x200px) (0-199 array index range)

        - Target Template Patch Row Index = 100

        - Scale Factor = 2.0

        - 100 * 2.0 = 200.00 (int: 200)

        - 200 > 199 max image index = ERROR

        """
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(template_patch_width, (scaled_search_window_width - 1))
        scale_factor_height = TSEGeometry.calc_measure_scale_factor(template_patch_height, (scaled_search_window_height - 1))

        scaled_window_heights = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_height, scale_factor_height)
        scaled_window_widths = TSEImageUtils.calc_scaled_image_pixel_dimension_coordinates(template_patch_width, scale_factor_width)

        # Extract all the pixels from the larger search window corresponding to the scaled coordinates of each pixel in the template patch.
        search_window_target_pixels = TSEImageUtils.extract_rows_cols_pixels_image(scaled_window_heights, scaled_window_widths, current_search_window)

        # Extract pixels from the larger current search window that are in the SCALED 2D-coordinates of the pixels in the original template patch.
        reshaped_search_window_target_pixels = TSEImageUtils.reshape_match_images(search_window_target_pixels, template_image)

        # Perform the actual template matching.
        res = cv2.matchTemplate(reshaped_search_window_target_pixels, template_image, match_method)

        # '_' is a placeholder convention to indicate we do not want to use these returned values.
        min_val, max_val, _, _ = cv2.minMaxLoc(res)

        # If we are matching using SSD, then the lowest score is the "best match", so return this.
        if match_method == cv2.cv.CV_TM_SQDIFF or match_method == cv2.cv.CV_TM_SQDIFF_NORMED:
            return min_val

        # Otherwise (and in most cases), we will want to return the highest score.
        return max_val


    @staticmethod
    def calc_compare_hsv_histogram(image_1, image_2, match_method):
        """
        Performs histogram-based template matching between two images represented by the HSV colour space, utilising the OpenCV 'calcHist' and 'compareHist' functions.

        :param image_1: The first image to be matched in the HSV colour space.
        :param image_2: The second image to be matched in the HSV colour space.
        :param match_method: Enum representing the similarity metric to be used ('CV_COMP_CHISQR' for Chi-Square test and 'CV_COMP_CCORR' for Histogram Correlation)
        :return Float representing the match score calculated by the histogram-based similarity metric.
        """

        # Calculate the histograms for both images. We only take notice of the Hue and Sat channels.
        hist_patch1 = cv2.calcHist([image_1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_patch2 = cv2.calcHist([image_2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # Perform the histogram-based matching on the two HSV colour histograms.
        hist_compare_result = cv2.compareHist(hist_patch1, hist_patch2, match_method)

        return hist_compare_result

    @staticmethod
    def convert_hsv_and_remove_luminance(image):
        """
        Converts an RGB colour image to use the HSV colour space (using 'cv2.cvtColor'), before removing the 'Value' channel by setting it to 0 in all pixels.

        :param image: The image to be converted.
        :return The same image converted to HSV colour space and with the 'Value' channel in effect removed.
        """

        # Convert the RGB image to HSV.
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Set the 'V' channel of each pixel in the image to '0' (i.e remove it)
        hsv_image[:, :, 2] = 0

        return hsv_image

    @staticmethod
    def scale_image_roi_relative_centre(origin_coordinate, end_coordinate, scale_factor):
        """
        Scales the boundary edge of an image relative to the centre coordinate, based upon a specific scale factor.

        :param origin_coordinate: The top-left corner coordinate of the image.
        :param end_coordinate: The bottom-right corner coordinate of the image.
        :param scale_factor: The factor from which to scale by.
        :return Tuple containing the scaled top-left and bottom-right origin coordinates.
        """

        height = end_coordinate[1] - origin_coordinate[1]
        width = end_coordinate[0] - origin_coordinate[0]

        centre = ((origin_coordinate[0] + int(round(width / 2.0))), (origin_coordinate[1] + int(round(height / 2.0))))

        # Calculate the scale vector and find it's length relative to the centre coordinate.
        scaled_origin = TSEGeometry.scale_coordinate_relative_centre(origin_coordinate, centre, scale_factor)

        scaled_end = TSEGeometry.scale_coordinate_relative_centre(end_coordinate, centre, scale_factor)

        return TSEPoint(scaled_origin[0], scaled_origin[1]), TSEPoint(scaled_end[0], scaled_end[1])

    @staticmethod
    def scale_image_no_interpolation_auto(source_image, target_image):
        """
        Geometrically scales each pixel of an image relative to the centre pixel without interpolating between original pixels.
        Calculates scale factor based upon the difference in size between two input images.

        :param source_image: The original image that we wish to scale.
        :param target_image: The image of whose dimensions we wish to scale to.
        :return New image containing the original pixels moved to their new scaled position.
        """

        source_image_height, source_image_width = source_image.shape[:2]
        current_image_height, current_image_width = target_image.shape[:2]

        scale_factor_height = TSEGeometry.calc_measure_scale_factor(source_image_height, current_image_height)
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(source_image_width, current_image_width)

        # Calculate what the dimensions of the scaled image should be.
        scaled_source_image_height = round(source_image_height * scale_factor_height)
        scaled_source_image_width = round(source_image_width * scale_factor_width)

        scaled_image_result = np.zeros((scaled_source_image_height, scaled_source_image_width, 3), np.uint8)

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(source_image_height):
            for j in xrange(source_image_width):

                template_patch_val_channel_1 = source_image.item(i, j, 0)
                template_patch_val_channel_2 = source_image.item(i, j, 1)
                template_patch_val_channel_3 = source_image.item(i, j, 2)

                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 0),
                                            template_patch_val_channel_1)
                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 1),
                                            template_patch_val_channel_2)
                scaled_image_result.itemset((i * scale_factor_height, (j * scale_factor_width), 2),
                                            template_patch_val_channel_3)

        return scaled_image_result

    @staticmethod
    def scale_image_interpolation_auto(source_image, target_image):
        """
        Scales the original image by INTERPOLATING between original pixels.
        Calculates scale factor based upon the difference in size between two input images. Utilises OpenCV 'resize' function.

        :param source_image: The original image that we wish to scale.
        :param target_image: The image of whose dimensions we wish to scale to.
        :return New image containing the original pixels moved to their new scaled position, with interpolated pixels in between the original ones.
        """
        source_image_height, source_image_width = source_image.shape[:2]
        current_image_height, current_image_width = target_image.shape[:2]

        scale_factor_height = TSEGeometry.calc_measure_scale_factor(source_image_height, current_image_height)
        scale_factor_width = TSEGeometry.calc_measure_scale_factor(source_image_width, current_image_width)

        scaled_source_image_height = round(source_image_height * scale_factor_height)
        scaled_source_image_width = round(source_image_width * scale_factor_width)

        # Calculate and define what the dimensions of the new image should be, before passing to the OpenCV function to resize.
        dim = (int(scaled_source_image_width), int(scaled_source_image_height))

        return cv2.resize(source_image, dim)

    @staticmethod
    def scale_image_interpolation_man(source_image, scale_factor):
        """
        Scales the original image by INTERPOLATING between original pixels. Utilises OpenCV 'resize' function.

        :param source_image: The original image that we wish to scale.
        :param scale_factor: The factor by which we wish to scale the original image.
        :return New image containing the original pixels moved to their new scaled position, with interpolated pixels in between the original ones.
        """

        source_image_height, source_image_width = source_image.shape[:2]

        dim = (int(source_image_width * scale_factor), int(source_image_height * scale_factor))

        return cv2.resize(source_image, dim)

    @staticmethod
    def extract_image_sub_window(source_image, origin_coordinates, end_coordinates):
        """
        Extracts a region from the original image defined by top-left and bottom-right origin coordinates.

        :param source_image: The image from which a region is to be extracted.
        :param origin_coordinate: The top-left corner coordinate of the region boundary.
        :param end_coordinate: The bottom-right corner coordinate of the region boundary.
        :return New image containing the portion of the orignal image within the bounding box.
        """
        # Numoy allows you to specify a range of indices to extract from a multi-dimensional array.
        return source_image[origin_coordinates.y:end_coordinates.y, origin_coordinates.x: end_coordinates.x]

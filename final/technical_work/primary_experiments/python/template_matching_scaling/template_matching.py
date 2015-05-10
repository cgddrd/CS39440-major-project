"""

Module Name: TemplateMatching

Description: Performs each template matching experiment based upon the parameters passed in by the test rig.

"""

from __future__ import division

__author__ = 'Connor Luke Goddard (clg11)'

import cv2

from tse.tse_fileio import TSEFileIO
from tse.tse_datautils import TSEDataUtils
from tse.tse_point import TSEPoint
from tse.tse_imageutils import TSEImageUtils
from tse.tse_result import TSEResult
from tse.tse_geometry import TSEGeometry
from tse.tse_matchmethod import tse_match_methods


class TemplateMatching:
    def __init__(self, image_one_file_path, image_two_file_path, calibration_data_file_path, plot_axis, use_hsv_colour_space=True, strip_luminance=True):

        self._image_one_file_path = image_one_file_path
        self._image_two_file_path = image_two_file_path

        # Find the last instance of '/' in the file path, and grab the image name from the split array.
        self._image_one_file_name = image_one_file_path.rsplit('/', 1)[1]
        self._image_two_file_name = image_two_file_path.rsplit('/', 1)[1]

        self._img1_raw = cv2.imread(image_one_file_path, cv2.IMREAD_COLOR)
        self._img2_raw = cv2.imread(image_two_file_path, cv2.IMREAD_COLOR)

        if use_hsv_colour_space:

            if strip_luminance:
                self._img1_target_color_space = TSEImageUtils.convert_hsv_and_remove_luminance(self._img1_raw)
                self._img2_target_colour_space = TSEImageUtils.convert_hsv_and_remove_luminance(self._img2_raw)

            else:
                self._img1_target_color_space = cv2.cvtColor(self._img1_raw, cv2.COLOR_BGR2HSV)
                self._img2_target_colour_space = cv2.cvtColor(self._img2_raw, cv2.COLOR_BGR2HSV)

        else:

            self._img1_target_color_space = self._img1_raw.copy()
            self._img2_target_colour_space = self._img2_raw.copy()

        self._calibration_lookup_table = self.load_calibration_data(calibration_data_file_path)
        self._calibration_data_file_path = calibration_data_file_path

        self._plot_axis = plot_axis

    def load_calibration_data(self, file_path):

        raw_data = TSEFileIO.read_file(file_path, split_delimiter=",", start_position=1)
        return dict(TSEDataUtils.string_2d_list_to_int_2d_list(raw_data))

    def run_template_search(self, patch_height, match_method, use_scaling=False, scale_centre=False, exhaustive_search=False, plot_results=False):
        """
        Moves through each image row, performing template matching search down through a localised search column originating from the current row to
        the bottom of the image.

        :param patch_height: The fixed height of the full-width patch.
        :param match_method: The current similarity metric for use in locating the best match within the localised search window.
        :param use_scaling: Specifies if the tests should perform geometric scaling (i.e. EXPERIMENT 3 method) or not (EXPERIMENT 2 method).
        :param scale_centre: Specifies whether scaling of pixel coordinates should be calculated relative to the centre, or top of the image.
        :param exhaustive_search: Specifies if the tests should perform exhaustive or non-exhaustive searching.
        :param plot_results: Specifies if the results from each test should be graphically plotted or not.
        :return: Dictionary structure containing the results for all experiment tests conducted.
        """
        run_results = []

        # Identify the starting row defined within the calibration data file for accounting for perspective distortion.
        smallest_key = TSEDataUtils.get_smallest_key_dict(self._calibration_lookup_table)

        image_height, image_width = self._img2_target_colour_space.shape[:2]

        image_centre_x = int(round(image_width / 2.0))

        # Main loop moving down EVERY ROW IN THE IMAGE that we wish to then perform a localised search for the BEST match between two images.
        for i in xrange(smallest_key, image_height - patch_height):

            calibrated_patch_width = self._calibration_lookup_table[i]

            patch_half_width = int(round(calibrated_patch_width / 2.0))

            # Create points for the current template patch origin, end and centre.
            template_patch_origin_point = TSEPoint((image_centre_x - patch_half_width), i)
            template_patch_end_point = TSEPoint((image_centre_x + patch_half_width), (i + patch_height))

            template_patch = self._img1_target_color_space[template_patch_origin_point.y: template_patch_end_point.y, template_patch_origin_point.x: template_patch_end_point.x]

            if use_scaling is True:

                if scale_centre:

                    # We are wanting to scale the template patch relative to its centre - RUN THE LOCALISED SEARCH!
                    displacement, scores = self.template_matching_roi_scale_centre(template_patch, template_patch_origin_point, match_method, exhaustive_search)
                else:

                    # We are wanting to scale the template patch relative to its top - RUN THE LOCALISED SEARCH!
                    displacement, scores = self.template_matching_roi_scale_top(template_patch, template_patch_origin_point, match_method, exhaustive_search)

                # Append the results of the current search to the collection of results for the entire experiment test.
                run_results.append(TSEResult(i, displacement, scores))

            else:

                # Otherwise, we do not want to scale - RUN THE LOCALISED SEARCH!
                displacement, scores = self.template_matching_roi_no_scale(template_patch, template_patch_origin_point, match_method, exhaustive_search)

                # Append the results of the current search to the collection of results for the entire experiment test.
                run_results.append(TSEResult(i, displacement, scores))

        # Once all of the tests have been completed...

        # If we want to plot the results to GUI, then do so.
        if plot_results:
            self.plot_results(run_results, match_method)

        # Return the results of the entire test back to the automated test rig so that it can add them to the complete set of results for all of the tests.
        return run_results

    def template_matching_roi_no_scale(self, template_patch, template_patch_origin, match_method, exhaustive_search=False):
        """
        Performs template matching using either an exhaustive or non-exhaustive search within a localised search column spanning from the current row
        to the bottom of the image. No scaling is performed.

        :param template_patch: The current template patch that we wish to try and locate in the second image.
        :param template_patch_origin: The Y-coordinate of the template origin used to define the starting Y-position of the localised search window.
        :param match_method: The current similarity metric for use in locating the best match within the localised search window.
        :param exhaustive_search: Specifies if the tests should perform exhaustive or non-exhaustive searching.
        :return: Tuple containing a collection of all recorded scores, in addition to the row number that provided the best match score.
        """

        image_height, image_width = self._img2_target_colour_space.shape[:2]

        template_patch_height, template_patch_width = template_patch.shape[:2]

        # Extract the localised search column from the second image. The height of this window spans from the origin of the template patch, and the width is equal to the width of the template patch.
        localised_window = self._img2_target_colour_space[template_patch_origin.y:image_height, template_patch_origin.x:(template_patch_origin.x + template_patch_width)]

        localised_window_height, localised_window_width = localised_window.shape[:2]

        best_score = -1
        best_position = 0
        scores = []

        stop = False

        # Move down through the localised search column one row/pixel at a time.
        for i in xrange(0, (localised_window_height - template_patch_height)):

            # Extract the current patch from within the localised search column to compare against the template patch for a best match.
            current_window = localised_window[i:(i + template_patch_height), 0:template_patch_width]

            score = 0

            # Calculate the similarity match between the two image patches using the specified similarity metric.
            if match_method.match_type == tse_match_methods.DISTANCE_ED:
                score = TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, current_window)

            elif match_method.match_type == tse_match_methods.DISTANCE:
                score = TSEImageUtils.calc_template_match_compare_cv2_score(template_patch, current_window, match_method.match_id)

            elif match_method.match_type == tse_match_methods.HIST:
                score = TSEImageUtils.calc_compare_hsv_histogram(template_patch, current_window, match_method.match_id)

            # If lower score means better match, then the method is a 'reverse' method.
            if match_method.reverse_score:

                # If the search has just begun, OR the current score is better than the recorded best score..
                if best_score == -1 or score < best_score:

                    # Set the current score as the new best score, and the current row number as the row with the best match.

                    """
                    NOTE: The recorded displacement is in effect the value of 'best_position' given that this value is
                    recorded in a search where the ORIGIN begins at the template patch.
                    """
                    best_score = score
                    best_position = i

                # Otherwise, we need to make a note to stop the search IN CASE we are performing a NON-EXHAUSTIVE search.
                else:
                    stop = True

            else:

                if best_score == -1 or score > best_score:
                    best_score = score
                    best_position = i

                else:
                    stop = True

            # Add the current score (regardless of its value) to the list of all recorded scores.
            scores.append((score, i))

            # If we are performing a non-exhaustive search, we may now want to stop the search at this point.
            if (exhaustive_search is False) and (stop is True):
                break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position, scores

    def template_matching_roi_scale_centre(self, template_patch, template_patch_origin, match_method, exhaustive_search=False):
        """
        Performs template matching using either an exhaustive or non-exhaustive search within a localised search column spanning from the current row
        to the bottom of the image. Geometric scaling of the template patch is performed relative to its centre.

        :param template_patch: The current template patch that we wish to try and locate in the second image.
        :param template_patch_origin: The Y-coordinate of the template origin used to define the starting Y-position of the localised search window.
        :param match_method: The current similarity metric for use in locating the best match within the localised search window.
        :param exhaustive_search: Specifies if the tests should perform exhaustive or non-exhaustive searching.
        :return: Tuple containing a collection of all recorded scores, in addition to the row number that provided the best match score.
        """

        image_height, image_width = self._img2_target_colour_space.shape[:2]

        image_centre_x = int(round(image_width / 2.0))

        template_patch_height, template_patch_width = template_patch.shape[:2]

        new_localised_window_height = image_height - template_patch_height

        best_score = -1
        best_position = 0
        scores = []

        stop = False

        last_width = template_patch_width

        prev_current_window_scaled_coords = None

        # Move down through the localised search column one row/pixel at a time.
        for i in xrange(template_patch_origin.y, new_localised_window_height):

            score = 0

            # If we have moved down by at least one row already, then we need to the value of the PREVIOUS calibrated width to calculate the new SCALE FACTOR.
            if i >= (template_patch_origin.y + 1):
                last_width = self._calibration_lookup_table[i - 1]

            # Obtain the new required full-width for the patch from the calibration file.
            calibrated_patch_width = self._calibration_lookup_table[i]

            patch_half_width = int(round(calibrated_patch_width / 2.0))

            # Calculate the scaling factor between the last ROI width, and the new ROI width.
            scale_factor = TSEGeometry.calc_measure_scale_factor(last_width, calibrated_patch_width)

            # For efficiency, store the previous scaled coordinates so that we do not have to re-calculate from scratch on each run.
            if prev_current_window_scaled_coords is None:

                # Locate the centre pixel coordinate that will define the origin of the geometric scaling procedure.
                current_window_scaled_coords = TSEPoint((image_centre_x - patch_half_width), i), TSEPoint((image_centre_x + patch_half_width), i + template_patch_height)

            else:

                # We add +1 to the 'Y' coordinate as we are moving the search window down the ROI by one pixel each time we increase the width.
                current_window_scaled_coords = TSEImageUtils.scale_image_roi_relative_centre(
                    (prev_current_window_scaled_coords[0].x, prev_current_window_scaled_coords[0].y + 1),
                    (prev_current_window_scaled_coords[1].x, prev_current_window_scaled_coords[1].y + 1),
                    scale_factor)

            prev_current_window_scaled_coords = current_window_scaled_coords

            # Extract pixels from the larger search window that correspond to the SCALED position of each pixel in the smaller template patch.
            scaled_search_window = TSEImageUtils.extract_image_sub_window(self._img2_target_colour_space, current_window_scaled_coords[0], current_window_scaled_coords[1])

            if match_method.match_type == tse_match_methods.DISTANCE_ED:
                score = TSEImageUtils.calc_ed_template_match_score_scaled_compiled(template_patch, scaled_search_window)

            elif match_method.match_type == tse_match_methods.DISTANCE:
                score = TSEImageUtils.calc_template_match_compare_cv2_score_scaled(template_patch, scaled_search_window, match_method.match_id)

            elif match_method.match_type == tse_match_methods.HIST:
                scaled_template_patch = TSEImageUtils.scale_image_interpolation_auto(template_patch, scaled_search_window)

                score = TSEImageUtils.calc_compare_hsv_histogram(scaled_template_patch, scaled_search_window, match_method.match_id)

            # If lower score means better match, then the method is a 'reverse' method.
            if match_method.reverse_score:

                if best_score == -1 or score < best_score:
                    best_score = score
                    best_position += 1

                else:
                    stop = True

            else:

                if best_score == -1 or score > best_score:
                    best_score = score
                    best_position += 1

                else:
                    stop = True

            scores.append((score, i))

            if (exhaustive_search is False) and (stop is True):
                break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position, scores

    def template_matching_roi_scale_top(self, template_patch, template_patch_origin, match_method, exhaustive_search=False):
        """
        Performs template matching using either an exhaustive or non-exhaustive search within a localised search column spanning from the current row
        to the bottom of the image. Geometric scaling of the template patch is performed relative to its top origin.

        :param template_patch: The current template patch that we wish to try and locate in the second image.
        :param template_patch_origin: The Y-coordinate of the template origin used to define the starting Y-position of the localised search window.
        :param match_method: The current similarity metric for use in locating the best match within the localised search window.
        :param exhaustive_search: Specifies if the tests should perform exhaustive or non-exhaustive searching.
        :return: Tuple containing a collection of all recorded scores, in addition to the row number that provided the best match score.
        """

        image_height, image_width = self._img2_target_colour_space.shape[:2]

        image_centre_x = int(round(image_width / 2.0))

        template_patch_height, template_patch_width = template_patch.shape[:2]

        new_localised_window_height = image_height - template_patch_height

        best_score = -1
        best_position = 0
        scores = []

        stop = False

        last_width = template_patch_width

        prev_current_window_scaled_coords = None

        prev_height = template_patch_height

        for i in xrange(template_patch_origin.y, new_localised_window_height):

            score = 0

            if i >= (template_patch_origin.y + 1):
                last_width = self._calibration_lookup_table[i - 1]

            calibrated_patch_width = self._calibration_lookup_table[i]
            patch_half_width = int(round(calibrated_patch_width / 2.0))

            scale_factor = TSEGeometry.calc_measure_scale_factor(last_width, calibrated_patch_width)

            # If we are going to scale the template patch PAST THE MAXIMUM HEIGHT OF THE IMAGE, then we need to stop.
            if (i + int(round(prev_height * scale_factor))) > image_height:
                break

            if prev_current_window_scaled_coords is None:
                current_window_scaled_coords = TSEPoint((image_centre_x - patch_half_width), i), TSEPoint((image_centre_x + patch_half_width), i + template_patch_height)

            else:

                prev_height = prev_current_window_scaled_coords[1].y - prev_current_window_scaled_coords[0].y

                # This is the line which really separates this function from 'template_matching_roi_scale_centre'.
                # Here we are scaling from the TOP of the template patch, rather than from the CENTRE.
                current_window_scaled_coords = TSEPoint((image_centre_x - patch_half_width), i), TSEPoint((image_centre_x + patch_half_width), i + int(round(prev_height * scale_factor)))

            prev_current_window_scaled_coords = current_window_scaled_coords

            scaled_search_window = TSEImageUtils.extract_image_sub_window(self._img2_target_colour_space, current_window_scaled_coords[0], current_window_scaled_coords[1])

            if match_method.match_type == tse_match_methods.DISTANCE_ED:
                score = TSEImageUtils.calc_ed_template_match_score_scaled_compiled(template_patch, scaled_search_window)

            elif match_method.match_type == tse_match_methods.DISTANCE:
                score = TSEImageUtils.calc_template_match_compare_cv2_score_scaled(template_patch, scaled_search_window, match_method.match_id)

            elif match_method.match_type == tse_match_methods.HIST:
                scaled_template_patch = TSEImageUtils.scale_image_interpolation_auto(template_patch, scaled_search_window)

                score = TSEImageUtils.calc_compare_hsv_histogram(scaled_template_patch, scaled_search_window, match_method.match_id)

            # If lower score means better match, then the method is a 'reverse' method.
            if match_method.reverse_score:

                if best_score == -1 or score < best_score:
                    best_score = score
                    best_position += 1

                else:
                    stop = True

            else:

                if best_score == -1 or score > best_score:
                    best_score = score
                    best_position += 1

                else:
                    stop = True

            scores.append((score, i))

            if (exhaustive_search is False) and (stop is True):
                break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position, scores

    def plot_results(self, results, match_method):
        """
        Plots the results of the current test to a new graph using the 'Matplotlib' library.

        :param results: The current template patch that we wish to try and locate in the second image.
        :param match_method: The current similarity metric for use in locating the best match within the localised search window.
        :return: Tuple containing a collection of all recorded scores, in addition to the row number that provided the best match score.
        """

        x = []
        y = []

        # Obtain the plot line format string.
        plot_format_color = match_method.format_string

        # Unpack all of the results ready to plot.
        for val in results:
            x.append(val.row)
            y.append(val.displacement)

        # Calculate the centered moving average for all of the displacement results.
        y_centered_moving_average = TSEDataUtils.calc_centered_moving_average(y, 10)


        # Plot both the original results, and the averaged results on the graph.
        self.plot(x, y, "{0}.".format(plot_format_color), match_method.match_name, 100)
        self.plot(x[len(x) - len(y_centered_moving_average):], y_centered_moving_average,
                  "{0}-".format(plot_format_color), "{0}_CMA".format(match_method.match_name), 100)

    def plot(self, data_x, data_y, plot_format, plot_name, max_boundary_offset=0):
        """
        Sets up the graph for the current test, and plots the results via the 'Matplotlib' library.

        :param data_x: The data to be plotted along the X-axis (i.e. the row number)
        :param data_y: The data to be plotted along the Y-axis (i.e. the level of recorded vertical displacement for a given row)
        :param plot_format: The format string used by the 'Matplotlib' library to apply properties such as colour and line thickness.
        :param plot_name: A representative name for the current results in order to identify via the graph legend.
        :param max_boundary_offset: The offset that should be applied to the X-axis in order to focus upon a particular subset of results.
        """

        self._plot_axis.set_xlim(0, data_x[(len(data_x) - 1)] + max_boundary_offset)

        self._plot_axis.grid(True)

        # Actually plot the data and re-draw the single figure.
        self._plot_axis.plot(data_x, data_y, plot_format, label=plot_name)

        self._plot_axis.legend(loc='upper left', shadow=True)

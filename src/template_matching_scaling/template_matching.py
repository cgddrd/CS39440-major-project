from __future__ import division

import cv2

from tse.tse_fileio import TSEFileIO
from tse.tse_datautils import TSEDataUtils
from tse.tse_point import TSEPoint
from tse.tse_imageutils import TSEImageUtils
from tse.tse_result import TSEResult
from tse.tse_geometry import TSEGeometry
from tse.tse_matchmethod import tse_match_methods

__author__ = 'connorgoddard'


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

        run_results = []

        smallest_key = TSEDataUtils.get_smallest_key_dict(self._calibration_lookup_table)

        image_height, image_width = self._img2_target_colour_space.shape[:2]

        image_centre_x = int(round(image_width / 2.0))

        for i in xrange(smallest_key, image_height - patch_height):

            calibrated_patch_width = self._calibration_lookup_table[i]

            patch_half_width = int(round(calibrated_patch_width / 2.0))

            # Create points for the current template patch origin, end and centre.
            template_patch_origin_point = TSEPoint((image_centre_x - patch_half_width), i)
            template_patch_end_point = TSEPoint((image_centre_x + patch_half_width), (i + patch_height))

            template_patch = self._img1_target_color_space[template_patch_origin_point.y: template_patch_end_point.y, template_patch_origin_point.x: template_patch_end_point.x]

            if use_scaling is True:

                if scale_centre:
                    displacement, scores = self.template_matching_roi_scale_centre(template_patch, template_patch_origin_point, match_method, exhaustive_search)
                else:
                    displacement, scores = self.template_matching_roi_scale_top(template_patch, template_patch_origin_point, match_method, exhaustive_search)

                run_results.append(TSEResult(i, displacement, scores))

            else:

                displacement, scores = self.template_matching_roi_no_scale(template_patch, template_patch_origin_point, match_method, exhaustive_search)
                run_results.append(TSEResult(i, displacement, scores))

        if plot_results:
            # self._plot_axis.set_xlabel('Row Number (px)')
            # self._plot_axis.set_ylabel('Vertical Displacement (px)')
            # self._plot_axis.set_title('Patch: {0}px - Images: {1}, {2}'.format(patch_height, self._image_one_file_name, self._image_two_file_name))
            self.plot_results(run_results, match_method)

        return run_results

    def template_matching_roi_no_scale(self, template_patch, template_patch_origin, match_method, exhaustive_search=False):

        image_height, image_width = self._img2_target_colour_space.shape[:2]

        template_patch_height, template_patch_width = template_patch.shape[:2]

        localised_window = self._img2_target_colour_space[template_patch_origin.y:image_height, template_patch_origin.x:(template_patch_origin.x + template_patch_width)]

        localised_window_height, localised_window_width = localised_window.shape[:2]

        best_score = -1
        best_position = 0
        scores = []

        stop = False

        for i in xrange(0, (localised_window_height - template_patch_height)):

            current_window = localised_window[i:(i + template_patch_height), 0:template_patch_width]
            score = 0

            if match_method.match_type == tse_match_methods.DISTANCE_ED:
                score = TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, current_window)

            elif match_method.match_type == tse_match_methods.DISTANCE:
                score = TSEImageUtils.calc_template_match_compare_cv2_score(template_patch, current_window, match_method.match_id)

            elif match_method.match_type == tse_match_methods.HIST:
                score = TSEImageUtils.calc_compare_hsv_histogram(template_patch, current_window, match_method.match_id)

            # If lower score means better match, then the method is a 'reverse' method.
            if match_method.reverse_score:

                if best_score == -1 or score < best_score:
                    best_score = score
                    best_position = i

                else:
                    stop = True

            else:

                if best_score == -1 or score > best_score:
                    best_score = score
                    best_position = i

                else:
                    stop = True

            scores.append((score, i))

            if (exhaustive_search is False) and (stop is True):
                break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position, scores

    def template_matching_roi_scale_centre(self, template_patch, template_patch_origin, match_method, exhaustive_search=False):

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

        for i in xrange(template_patch_origin.y, new_localised_window_height):

            score = 0

            if i >= (template_patch_origin.y + 1):
                last_width = self._calibration_lookup_table[i - 1]

            calibrated_patch_width = self._calibration_lookup_table[i]
            patch_half_width = int(round(calibrated_patch_width / 2.0))

            scale_factor = TSEGeometry.calc_measure_scale_factor(last_width, calibrated_patch_width)

            if prev_current_window_scaled_coords is None:
                current_window_scaled_coords = TSEPoint((image_centre_x - patch_half_width), i), TSEPoint((image_centre_x + patch_half_width), i + template_patch_height)

            else:

                # We add +1 to the 'Y' coordinate as we are moving the search window down the ROI by one pixel each time we increase the width.
                current_window_scaled_coords = TSEImageUtils.scale_image_roi_relative_centre(
                    (prev_current_window_scaled_coords[0].x, prev_current_window_scaled_coords[0].y + 1),
                    (prev_current_window_scaled_coords[1].x, prev_current_window_scaled_coords[1].y + 1),
                    scale_factor)

            prev_current_window_scaled_coords = current_window_scaled_coords

            scaled_search_window = TSEImageUtils.extract_image_sub_window(self._img2_target_colour_space, current_window_scaled_coords[0], current_window_scaled_coords[1])

            # cv2.imshow("search window centered", scaled_search_window)
            # cv2.waitKey(100)

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

            if (i + int(round(prev_height * scale_factor))) > image_height:
                break

            if prev_current_window_scaled_coords is None:

                current_window_scaled_coords = TSEPoint((image_centre_x - patch_half_width), i), TSEPoint((image_centre_x + patch_half_width), i + template_patch_height)

            else:

                prev_height = prev_current_window_scaled_coords[1].y - prev_current_window_scaled_coords[0].y

                current_window_scaled_coords = TSEPoint((image_centre_x - patch_half_width), i), TSEPoint((image_centre_x + patch_half_width), i + int(round(prev_height * scale_factor)))

            prev_current_window_scaled_coords = current_window_scaled_coords

            scaled_search_window = TSEImageUtils.extract_image_sub_window(self._img2_target_colour_space, current_window_scaled_coords[0], current_window_scaled_coords[1])

            # cv2.imshow("search window", scaled_search_window)
            # cv2.waitKey(100)

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

        x = []
        y = []

        plot_format_color = match_method.format_string

        for val in results:
            x.append(val.row)
            y.append(val.displacement)

        y_centered_moving_average = TSEDataUtils.calc_centered_moving_average(y, 10)

        self.plot(x, y, "{0}.".format(plot_format_color), match_method.match_name, 100)
        self.plot(x[len(x) - len(y_centered_moving_average):], y_centered_moving_average,
                  "{0}-".format(plot_format_color), "{0}_CMA".format(match_method.match_name), 100)

    def plot(self, data_x, data_y, plot_format, plot_name, max_boundary_offset=0):

        self._plot_axis.set_xlim(0, data_x[(len(data_x) - 1)] + max_boundary_offset)

        self._plot_axis.grid(True)

        # Actually plot the data and re-draw the single figure.
        self._plot_axis.plot(data_x, data_y, plot_format, label=plot_name)

        self._plot_axis.legend(loc='upper left', shadow=True)

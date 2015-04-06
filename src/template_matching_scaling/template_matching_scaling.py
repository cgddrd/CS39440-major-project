from __future__ import division

import cv2
import math
import matplotlib.pyplot as plt
import argparse

from tse.tse_fileio import TSEFileIO
from tse.tse_utils import TSEUtils
from tse.tse_point import TSEPoint
from tse.tse_imageutils import TSEImageUtils
from tse.tse_result import TSEResult
from tse.tse_geometry import TSEGeometry
from tse.tse_matchtype import TSEMatchType
from tse.tse_matchmethod import tse_match_methods

from collections import OrderedDict

__author__ = 'connorgoddard'


class TemplateMatching:
    def __init__(self, image_one_file_path, image_two_file_path, calib_data_file_path, plot_axis):

        self._image_one_file_path = image_one_file_path
        self._image_two_file_path = image_two_file_path

        # Find the last instance of '/' in the file path, and grab the image name from the split array.
        self._image_one_file_name = image_one_file_path.rsplit('/', 1)[1]
        self._image_two_file_name = image_two_file_path.rsplit('/', 1)[1]

        self._raw_img1 = cv2.imread(image_one_file_path, cv2.IMREAD_COLOR)
        self._raw_img2 = cv2.imread(image_two_file_path, cv2.IMREAD_COLOR)

        self._hsv_img1 = TSEImageUtils.convert_hsv_and_remove_luminance(self._raw_img1)
        self._hsv_img2 = TSEImageUtils.convert_hsv_and_remove_luminance(self._raw_img2)

        self._calibration_lookup = self.load_calibration_data(calib_data_file_path)
        self._calib_data_file_path = calib_data_file_path

        self._plot_axis = plot_axis

    def load_calibration_data(self, file_path):

        raw_data = TSEFileIO.read_file(file_path, split_delimiter=",", start_position=1)
        return dict(TSEUtils.string_2d_list_to_int_2d_list(raw_data))

    def search_image(self, patch_height, match_method, use_scaling=False, force_cont_search=False, plot_results=False):

        run_results = []

        smallest_key = TSEUtils.get_smallest_key_dict(self._calibration_lookup)

        image_height, image_width = self._hsv_img2.shape[:2]

        image_centre_x = math.floor(image_width / 2)

        for i in range(smallest_key + 1, image_height - patch_height):

            calibrated_patch_width = self._calibration_lookup[i]
            patch_half_width = math.floor(calibrated_patch_width / 2)

            # Create points for the current template patch origin, end and centre.
            template_patch_origin_point = TSEPoint((image_centre_x - patch_half_width), i)
            template_patch_end_point = TSEPoint((image_centre_x + patch_half_width), (i + patch_height))

            template_patch = self._hsv_img1[template_patch_origin_point.y: template_patch_end_point.y, template_patch_origin_point.x: template_patch_end_point.x]

            if use_scaling is True:

                run_results.append(TSEResult(i, self.scan_search_window_scaling(template_patch, template_patch_origin_point, match_method, force_cont_search)))

            else:

                run_results.append(TSEResult(i, self.scan_search_window(template_patch, template_patch_origin_point, match_method, force_cont_search)))

        if plot_results:
            # self._plot_axis.set_xlabel('Row Number (px)')
            # self._plot_axis.set_ylabel('Vertical Displacement (px)')
            # self._plot_axis.set_title('Patch: {0}px - Images: {1}, {2}'.format(patch_height, self._image_one_file_name, self._image_two_file_name))
            self.plot_results(run_results, match_method)

        return run_results

    def scan_search_window(self, template_patch, template_patch_origin, match_method, force_cont_search=False):

        image_height, image_width = self._hsv_img2.shape[:2]

        template_patch_height, template_patch_width = template_patch.shape[:2]

        localised_window = self._hsv_img2[template_patch_origin.y:image_height, template_patch_origin.x:(template_patch_origin.x + template_patch_width)]

        localised_window_height, localised_window_width = localised_window.shape[:2]

        best_score = -1
        best_position = 0

        stop = False

        for i in range(0, (localised_window_height - template_patch_height)):

            current_window = localised_window[i:(i + template_patch_height), 0:template_patch_width]
            score = 0

            if match_method.match_type == tse_match_methods.DISTANCE_ED:
                score = TSEImageUtils.calc_euclidean_distance_cv2_norm(template_patch, current_window)

            elif match_method.match_type == tse_match_methods.DISTANCE:
                score = TSEImageUtils.calc_template_match_compare_cv2_score(template_patch, current_window,
                                                                            match_method.match_id)

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

            if (force_cont_search is False) and (stop is True):
                break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position

    def scan_search_window_scaling(self, template_patch, template_patch_origin, match_method, force_cont_search=False):

        image_height, image_width = self._hsv_img2.shape[:2]

        image_centre_x = math.floor(image_width / 2)

        template_patch_height, template_patch_width = template_patch.shape[:2]

        new_localised_window_height = image_height - template_patch_height

        best_score = -1
        best_position = 0

        stop = False

        last_width = template_patch_width

        prev_current_window_scaled_coords = None

        for i in range(template_patch_origin.y, new_localised_window_height):

            score = 0

            if i >= (template_patch_origin.y + 1):
                last_width = self._calibration_lookup[i - 1]

            calibrated_patch_width = self._calibration_lookup[i]
            patch_half_width = math.floor(calibrated_patch_width / 2)
            scale_factor = TSEGeometry.calc_measure_scale_factor(last_width, calibrated_patch_width)

            if prev_current_window_scaled_coords is None:

                current_window_scaled_coords = TSEImageUtils.scale_image_roi_relative_centre(
                    ((image_centre_x - patch_half_width), i),
                    ((image_centre_x + patch_half_width), (i + template_patch_height)), scale_factor)

            else:

                # We add +1 to the 'Y' coordinate as we are moving the search window down the ROI by one pixel each time we increase the width.
                current_window_scaled_coords = TSEImageUtils.scale_image_roi_relative_centre(
                    (prev_current_window_scaled_coords[0].x, prev_current_window_scaled_coords[0].y + 1),
                    (prev_current_window_scaled_coords[1].x, prev_current_window_scaled_coords[1].y + 1),
                    scale_factor)

            prev_current_window_scaled_coords = current_window_scaled_coords

            scaled_search_window = TSEImageUtils.extract_image_sub_window(self._hsv_img2, current_window_scaled_coords[0], current_window_scaled_coords[1])

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

            if (force_cont_search is False) and (stop is True):
                break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position

    def plot_results(self, results, match_method):

        x = []
        y = []

        plot_format_color = match_method.format_string

        for val in results:
            x.append(val.row)
            y.append(val.displacement)

        y_moving_average = TSEUtils.calc_moving_average_array(y, 10)

        self.plot(x, y, "{0}.".format(plot_format_color), 100, match_method.match_name)
        self.plot(x[len(x) - len(y_moving_average):], y_moving_average, "{0}-".format(plot_format_color), 100, "MVAV_{0}".format(match_method.match_name))

    def plot(self, data_x, data_y, plot_format, max_boundary_offset, plot_name):

        self._plot_axis.set_xlim(0, data_x[(len(data_x) - 1)] + max_boundary_offset)

        self._plot_axis.grid(True)

        # Actually plot the data and re-draw the single figure.
        self._plot_axis.plot(data_x, data_y, plot_format, label=plot_name)

        self._plot_axis.legend(loc='upper left', shadow=True)


def start_tests(image_pairs, patch_sizes, match_methods, config_file, use_scaling=False, force_cont_search=False, plot_results=False):

    results = []

    if plot_results is False:

        for pair in image_pairs:

            match = TemplateMatching(pair[0], pair[1], config_file, None)

            for patch_size in patch_sizes:
                for match_method in match_methods:
                    results.append(match.search_image(patch_size, match_method, use_scaling, force_cont_search, plot_results))

    else:

        for pair in image_pairs:

            plot_count = len(image_pairs)

            column_max = 2

            row_max = int(math.ceil(plot_count / float(column_max)))

            fig, axes = plt.subplots(row_max, column_max)

            column_count = 0
            row_count = 0

            if row_max > 1:
                match = TemplateMatching(pair[0], pair[1], config_file, axes[row_count, column_count])
            else:
                match = TemplateMatching(pair[0], pair[1], config_file, axes[column_count])

            for patch_size in patch_sizes:

                if row_max > 1:
                    match._plot_axis = axes[row_count, column_count]
                else:
                    match._plot_axis = axes[column_count]

                match._plot_axis.set_xlabel('Row Number (px)')
                match._plot_axis.set_ylabel('Vertical Displacement (px)')
                match._plot_axis.set_title('Patch: {0}px - Images: {1}, {2}'.format(patch_size, match._image_one_file_name, match._image_two_file_name))

                for match_method in match_methods:

                    results.append(match.search_image(patch_size, match_method, use_scaling, force_cont_search, plot_results))

                if column_count == (column_max - 1):
                    row_count += 1
                    column_count = 0
                else:
                    column_count += 1

            # If we do not have an even number of graphs, then we need to remove the last blank one.
            if (plot_count % column_max) != 0:

                if row_max > 1:

                    axes[-1, -1].axis('off')

                else:

                    axes[-1].axis('off')

        plt.show()

    return results


def InputImagePair(raw_argument_string):

    try:
        x, y = raw_argument_string.split(',')

        # Strip out any whitespace either side of arguments.
        return x.strip(), y.strip()
    except:
        raise argparse.ArgumentTypeError("Image pairs expect format \"<image_1_path>, <image_2_path>\"")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--calibfile', help='Datafile containing the calibration data', dest="calib_file", required=True)
    parser.add_argument('-i', '--images', help="Images", dest="image_pairs", type=InputImagePair, nargs='+', required=True)
    parser.add_argument('-p', '--patches', nargs='+', dest="patch_sizes", type=int, required=True)
    parser.add_argument('-m', '--methods', nargs='+', dest="match_methods", type=str, required=True)
    parser.add_argument('-s', '--scaling', dest='scaling', action='store_true')
    parser.add_argument('-d', '--drawplot', dest='plot_results', action='store_true')
    parser.add_argument('-f', '--forcecontsearch', dest='force_cont_search', action='store_true')

    args = vars(parser.parse_args())

    match_methods = []

    # OrderedDict is used to remove any duplicates.
    for method in list(OrderedDict.fromkeys(args['match_methods'])):

        if method == "DistanceEuclidean":
            match_methods.append(TSEMatchType("DistanceEuclidean", tse_match_methods.DISTANCE_ED, None, "r", reverse_score=True))

        elif method == "DistanceCorr":
            match_methods.append(TSEMatchType("DistanceCorr", tse_match_methods.DISTANCE, cv2.cv.CV_TM_CCORR_NORMED, "b"))

        elif method == "HistCorrel":
            match_methods.append(TSEMatchType("HistCorrel", tse_match_methods.HIST, cv2.cv.CV_COMP_CORREL, "b"))

        elif method == "HistChiSqr":
            match_methods.append(TSEMatchType("HistChiSqr", tse_match_methods.HIST, cv2.cv.CV_COMP_CHISQR, "g", reverse_score=True))

        else:
            parser.error("Error: \"{0}\" is not a valid matching method option.\nSupported Methods: \'DistanceEuclidean\', \'DistanceCorr\', \'HistCorrel\', \'HistChiSqr\' ".format(method))

    # Start the tests using settings passed in as command-line arguments.
    start_tests(args['image_pairs'], list(OrderedDict.fromkeys(args['patch_sizes'])), match_methods, args['calib_file'], use_scaling=args['scaling'], force_cont_search=args['force_cont_search'], plot_results=args['plot_results'])

if __name__ == '__main__':
    main()
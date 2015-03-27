from __future__ import division

import cv2
import math

import matplotlib.pyplot as plt
import numpy as np

from tse.tse_fileio import TSEFileIO
from tse.tse_utils import TSEUtils
from tse.tse_point import TSEPoint
from tse.tse_imageutils import TSEImageUtils
from tse.tse_result import TSEResult
from tse.tse_match_methods import tse_match_methods
from tse.tse_geometry import TSEGeometry


__author__ = 'connorgoddard'


class MatchMethod:
    def __init__(self, match_name, match_type, match_id, format_string, reverse=False):
        self._match_name = match_name
        self._match_type = match_type
        self._match_id = match_id
        self._format_string = format_string
        self._reverse = reverse

    @property
    def match_name(self):
        return self._match_name

    @property
    def match_type(self):
        return self._match_type

    @property
    def match_id(self):
        return self._match_id

    @property
    def format_string(self):
        return self._format_string

    @property
    def reverse(self):
        return self._reverse


class TemplateMatching:
    def __init__(self, image_root_file_path, image_one_file_name, image_two_file_name, calib_data_file_path, plot_axis):

        self._image_root_file_patch = image_root_file_path

        self._image_one_file_name = image_one_file_name
        self._image_two_file_name = image_two_file_name

        self._raw_img1 = cv2.imread("{0}/{1}".format(image_root_file_path, image_one_file_name), cv2.IMREAD_COLOR)
        self._raw_img2 = cv2.imread("{0}/{1}".format(image_root_file_path, image_two_file_name), cv2.IMREAD_COLOR)

        self._hsv_img1 = TSEImageUtils.convert_hsv_and_remove_luminance(self._raw_img1)
        self._hsv_img2 = TSEImageUtils.convert_hsv_and_remove_luminance(self._raw_img2)

        self._calibration_lookup = self.load_calibration_data(calib_data_file_path)
        self._calib_data_file_path = calib_data_file_path

        self._plot_axis = plot_axis

    def load_calibration_data(self, file_path):
        raw_data = TSEFileIO.read_file(file_path, split_delimiter=",", start_position=1)
        return dict(TSEUtils.string_list_to_int_list(raw_data))

    def search_image(self, patch_height, match_methods):

        smallest_key = TSEUtils.get_smallest_key_dict(self._calibration_lookup)

        image_height, image_width = self._hsv_img2.shape[:2]

        image_centre_x = math.floor(image_width / 2)

        self._plot_axis.set_xlabel('Row Number (px)')
        self._plot_axis.set_ylabel('Vertical Displacement (px)')
        self._plot_axis.set_title('Patch: {0}px - Images: {1}, {2}'.format(patch_height, self._image_one_file_name,
                                                                           self._image_two_file_name))
        for match_method in match_methods:

            results = []

            for i in range(smallest_key + 1, image_height - patch_height):

                calibrated_patch_width = self._calibration_lookup[i]
                patch_half_width = math.floor(calibrated_patch_width / 2)

                # Create points for the current template patch origin, end and centre.
                template_patch_origin_point = TSEPoint((image_centre_x - patch_half_width), i)
                template_patch_end_point = TSEPoint((image_centre_x + patch_half_width), (i + patch_height))

                template_patch = self._hsv_img1[template_patch_origin_point.y: template_patch_end_point.y, template_patch_origin_point.x: template_patch_end_point.x]

                results.append(TSEResult(i, self.scan_search_window(template_patch, template_patch_origin_point, match_method)))

            self.plot_results(results, match_method)


    def scan_search_window(self, template_patch, template_patch_origin, match_method):

        image_height, image_width = self._hsv_img2.shape[:2]

        image_centre_x = math.floor(image_width / 2)

        template_patch_height, template_patch_width = template_patch.shape[:2]

        # localised_window = self._hsv_img2[template_patch_origin.y:image_height,
        #                    template_patch_origin.x:(template_patch_origin.x + template_patch_width)]

        # localised_window_height, localised_window_width = localised_window.shape[:2]

        new_localised_window_height = image_height - template_patch_height

        best_score = -1
        best_position = 0

        stop = False

        last_width = template_patch_width

        prev_current_window_scaled_coords = None

        # NEED TO SCALE BOTH THE WIDTH AND HEIGHT!

        # for i in range(0, (localised_window_height - template_patch_height)):
        for i in range(template_patch_origin.y, new_localised_window_height):

            # if i == template_patch_origin.y + 2:
                 # cv2.waitKey()
                 # raise SystemExit

            if i >= (template_patch_origin.y + 1):
                last_width = self._calibration_lookup[i - 1]

            calibrated_patch_width = self._calibration_lookup[i]
            patch_half_width = math.floor(calibrated_patch_width / 2)

            print "Last Width: {0}, Current Width: {1}".format(last_width, calibrated_patch_width)

            scale_factor = TSEGeometry.calc_patch_scale_factor(last_width, calibrated_patch_width)

            print "Scale Factor: {0}".format(scale_factor)

            current_window_scaled_coords = None

            if prev_current_window_scaled_coords == None:

                current_window_scaled_coords = TSEGeometry.scale_patch(((image_centre_x - patch_half_width), i), ((image_centre_x + patch_half_width), (i + template_patch_height)), scale_factor)

            else:

                current_window_scaled_coords = TSEGeometry.scale_patch((prev_current_window_scaled_coords[0][0], prev_current_window_scaled_coords[0][1] + 1), (prev_current_window_scaled_coords[1][0], prev_current_window_scaled_coords[1][1] + 1), scale_factor)


            prev_current_window_scaled_coords = current_window_scaled_coords

            scaled_origin = current_window_scaled_coords[0]

            scaled_end = current_window_scaled_coords[1]

            current_window = self._hsv_img2[scaled_origin[1]:scaled_end[1], scaled_origin[0]: scaled_end[0]]

            score = 0

            self.scale_template(template_patch, current_window)

            cv2.imshow("current search", current_window)
            cv2.imshow("template patch", template_patch)
            # cv2.imshow("image1", self._hsv_img1)
            # cv2.imshow("image2", self._hsv_img2)

            print "Image Height: {0}, Total Localised Window Height: {1}, Template Origin Y: {3}, Current Height: {2}".format(image_height, new_localised_window_height, i, template_patch_origin.y)

            cv2.waitKey(100)

            # if match_method.match_type == tse_match_methods.DISTANCE_ED:
            #     score = TSEImageUtils.calc_euclidean_distance_norm(template_patch, current_window)
            #
            # elif match_method.match_type == tse_match_methods.DISTANCE:
            #     score = TSEImageUtils.calc_match_compare(template_patch, current_window, match_method.match_id)
            #
            # elif match_method.match_type == tse_match_methods.HIST:
            #     score = TSEImageUtils.hist_compare(template_patch, current_window, match_method.match_id)
            #
            # # If higher score means better match, then the method is a 'reverse' method.
            # if match_method.reverse:
            #
            #     if best_score == -1 or score > best_score:
            #         best_score = score
            #         best_position = i
            #
            #     else:
            #         stop = True
            #
            # else:
            #
            #     if best_score == -1 or score < best_score:
            #         best_score = score
            #         best_position = i
            #
            #     else:
            #         stop = True
            #
            # if stop:
            #     break

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position

    def scale_template(self, template_patch, current_window):

        template_patch_width = template_patch.shape[1]
        template_patch_height = template_patch.shape[0]
        current_window_width = current_window.shape[1]

        current_window_height = current_window.shape[0]

        # current_window_width = 200
        # current_window_height = 99
        #
        # # current_window_width = 400
        # # current_window_height = 198
        #
        # # current_window_width = 800
        # # current_window_height = 396

        template_patch_scale_factor = TSEGeometry.calc_patch_scale_factor(template_patch_width, current_window_width)

        template_patch_centre = ((0 + (template_patch_width/ 2)), (0 + (template_patch_height / 2)))

        print "Template Patch: {0}px x {1}px".format(template_patch_width, template_patch_height)
        print "Current Window: {0}px x {1}px".format(current_window_width, current_window_height)

        scaled1 = TSEGeometry.scale_coordinate_relative_centre((0, 0), template_patch_centre, template_patch_scale_factor)

        scaled2 = TSEGeometry.scale_coordinate_relative_centre((template_patch_width, template_patch_height), template_patch_centre, template_patch_scale_factor)


        scaled1 = ((0 * template_patch_scale_factor), (0 * template_patch_scale_factor))
        scaled2 = (int(template_patch_width * template_patch_scale_factor), int(template_patch_height * template_patch_scale_factor))

        print "Template Patch Scaled (0,0): {0}".format(scaled1)
        print "Template Patch Scaled (END,END): {0}".format(scaled2)

        image = np.zeros((scaled2[1], scaled2[0], 3), np.uint8)

        # Loop through each pixel in the template patch, and scale it in the larger scaled image.
        for i in xrange(template_patch_height):
            for j in xrange(template_patch_width):

                # OLD/SLOWER APPROACH USING DIRECT ARRAY ACCESS
                # template_patch_val = template_patch[i, j]
                # image[(i * template_patch_scale_factor), (j * template_patch_scale_factor)] = template_patch_val

                # FASTER METHOD USING THE NUMPY METHODS.
                # WE DON'T BOTHER DOING ANYTHING WITH THE 'V' CHANNEL, AS WE ARE IGNORING IT ANYWAY.
                template_patch_val_hue = template_patch.item(i, j, 0)
                template_patch_val_sat = template_patch.item(i, j, 1)
                # template_patch_val_val = template_patch.item(i, j, 2)

                image.itemset((i * template_patch_scale_factor, (j * template_patch_scale_factor), 0), template_patch_val_hue)
                image.itemset((i * template_patch_scale_factor, (j * template_patch_scale_factor), 1), template_patch_val_sat)
                # image.itemset((i * template_patch_scale_factor, (j * template_patch_scale_factor), 2), template_patch_val_val)

        cv2.imshow("ORIGINAL TEMPLATE PATCH", template_patch)

        cv2.imshow("SCALED TEMPLATE PATCH", image)

        # cv2.waitKey()

        r = current_window_width / template_patch_width
        dim = (current_window_width, int(template_patch.shape[0] * r))

        resized = cv2.resize(template_patch, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("resized", resized)

    def plot_results(self, results, match_method):

        x = []
        y = []

        plot_format_color = match_method.format_string

        for val in results:
            x.append(val.row)
            y.append(val.displacement)

        y_moving_average = TSEUtils.calc_moving_average(y, 10)

        self.plot(x, y, "{0}.".format(plot_format_color), 100, match_method.match_name)
        self.plot(x[len(x) - len(y_moving_average):], y_moving_average, "{0}-".format(plot_format_color), 100,
                  "MVAV_{0}".format(match_method.match_name))

    def plot(self, data_x, data_y, plot_format, max_boundary_offset, plot_name):

        # self._plot_axis.xlim(0, data_x[(len(data_x) - 1)] + max_boundary_offset)
        self._plot_axis.grid(True)

        # Actually plot the data and re-draw the single figure.
        self._plot_axis.plot(data_x, data_y, plot_format, label=plot_name)

        self._plot_axis.legend(loc='upper left', shadow=True)


def start_tests(image_path, image_pairs, patch_sizes, match_types, config_file):
    for patch_size in patch_sizes:

        plot_count = len(image_pairs)

        column_max = 2

        row_max = int(math.ceil(plot_count / float(column_max)))

        fig, axes = plt.subplots(row_max, column_max)

        column_count = 0
        row_count = 0

        for pair in image_pairs:

            if row_max > 1:
                match = TemplateMatching(image_path, pair[0], pair[1], config_file, axes[row_count, column_count])

            else:
                match = TemplateMatching(image_path, pair[0], pair[1], config_file, axes[column_count])

            match.search_image(patch_size, match_types)

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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("calib_data_file", help="the file containing the calibration data")
    # parser.add_argument("input_image_1", help="the first image")
    # parser.add_argument("input_image_2", help="the second image")
    # args = parser.parse_args()

    image_path = "../eval_data/motion_images/flat_10cm"

    config_file = "../perspective_calibration/data/calibdata_23_03_15_11_07_04.txt"

    image_pairs = [("IMG1.JPG", "IMG2.JPG")]

    patch_sizes = [50]

    match_method1 = MatchMethod("DistanceEuclidean", tse_match_methods.DISTANCE_ED, None, "r")
    match_method2 = MatchMethod("HistCorrel", tse_match_methods.HIST, cv2.cv.CV_COMP_CORREL, "b", True)
    match_method3 = MatchMethod("HistChiSqr", tse_match_methods.HIST, cv2.cv.CV_COMP_CHISQR, "g")

    match_methods = [match_method1]

    start_tests(image_path, image_pairs, patch_sizes, match_methods, config_file)


if __name__ == '__main__':  # if the function is the main function ...
    main()
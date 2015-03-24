from __future__ import division

import cv2

import argparse
import matplotlib.pyplot as plt
import math

from tsefileio import TSEFileIO
from tseutils import TSEUtils
from tsegeometry import TSEGeometry
from tsepoint import TSEPoint
from tseimageutils import TSEImageUtils


__author__ = 'connorgoddard'


class TemplateMatching:
    def __init__(self, image_one_file_path, image_two_file_path, calib_data_file_path):
        self._raw_img1 = cv2.imread(image_one_file_path, cv2.IMREAD_COLOR)
        self._raw_img2 = cv2.imread(image_two_file_path, cv2.IMREAD_COLOR)

        self._hsv_img1 = self.convert_hsv_and_remove_luminance(self._raw_img1)
        self._hsv_img2 = self.convert_hsv_and_remove_luminance(self._raw_img2)

        self._calibration_lookup = self.load_calibration_data(calib_data_file_path)
        self._calib_data_file_path = calib_data_file_path

    def load_calibration_data(self, file_path):
        raw_data = TSEFileIO.read_file(file_path, split_delimiter=",", start_position=1)
        return dict(TSEUtils.string_list_to_int_list(raw_data))

    def setup_image_gui(self, image):
        plt.figure()
        plt.axis("off")
        canvas = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def convert_hsv_and_remove_luminance(self, source_image):
        hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)

        # Set the 'V' channel of each pixel in the image to '0' (i.e remove it)
        hsv_image[:, :, 2] = 0

        return hsv_image

    def scan_search_window(self, template_patch, template_patch_origin):

        image_height, image_width = self._hsv_img2.shape[:2]

        template_patch_height, template_patch_width = template_patch.shape[:2]

        localised_window = self._hsv_img2[template_patch_origin.y:image_height, template_patch_origin.x:(template_patch_origin.x + template_patch_width)]

        localised_window_height, localised_window_width = localised_window.shape[:2]

        best_score = -1
        best_position = 0

        for i in range(0, (localised_window_height - template_patch_height)):

            current_window = localised_window[i:(i + template_patch_height), 0:template_patch_width]

            score = TSEImageUtils.calc_euclidean_distance_norm(template_patch, current_window)

            if best_score == -1 or score < best_score:
                best_score = score
                best_position = i

        # print "Best Y: {0} - Score: {1}".format(best_position, best_score)

        # We need to return the 'Y' with the best score (i.e. the displacement)
        return best_position

    def search_image(self, patch_height):

        smallest_key = TSEUtils.get_smallest_key_dict(self._calibration_lookup)

        image_height, image_width = self._hsv_img2.shape[:2]

        image_centre_x = math.floor(image_width / 2)

        for i in range(smallest_key, image_height - patch_height):

            if i == (smallest_key + 1):
                break

            calibrated_patch_width = self._calibration_lookup[i]
            patch_half_width = math.floor(calibrated_patch_width / 2)

            patch_origin_x = (image_centre_x - patch_half_width)
            patch_end_x = (image_centre_x + patch_half_width)
            patch_origin_y = i
            patch_end_y = (i + patch_height)

            patch_origin_xy = (int(patch_origin_x), int(patch_origin_y))
            patch_end_xy = (int(patch_end_x), int(patch_end_y))

            patch_centre = (image_centre_x, (patch_origin_y + int(patch_height / 2)))

            patch_origin_xy_scaled = TSEGeometry.scale_coordinate_relative_centre(patch_origin_xy, patch_centre, 2)
            patch_end_xy_scaled = TSEGeometry.scale_coordinate_relative_centre(patch_end_xy, patch_centre, 2)

            template_patch = self._hsv_img1[patch_origin_y:patch_end_y, patch_origin_x:patch_end_x]

            template_patch_origin = TSEPoint(patch_origin_x, patch_origin_y)

            cv2.imshow("ROI", template_patch)

            best_position = self.scan_search_window(template_patch, template_patch_origin)

            cv2.rectangle(self._hsv_img2, (int(patch_origin_x), patch_origin_y + best_position), (int(patch_origin_x + calibrated_patch_width), int(patch_origin_y + best_position + patch_height)), (0, 0, 255), 2)

            cv2.imshow("bdsadsa", self._hsv_img2)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("calib_data_file", help="the file containing the calibration data")
    parser.add_argument("input_image_1", help="the first image")
    parser.add_argument("input_image_2", help="the second image")

    args = parser.parse_args()

    match = TemplateMatching(args.input_image_1, args.input_image_2, args.calib_data_file)

    match.search_image(40)

    cv2.waitKey()

if __name__ == '__main__':  # if the function is the main function ...
    main()
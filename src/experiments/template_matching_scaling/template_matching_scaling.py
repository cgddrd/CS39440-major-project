from __future__ import division

import cv2

import argparse
import matplotlib.pyplot as plt
import math

from tsefileio import TSEFileIO
from tseutils import TSEUtils
from tsegeometry import TSEGeometry


__author__ = 'connorgoddard'


class TemplateMatching:
    def __init__(self, image_one_file_path, image_two_file_path, calib_data_file_path):
        self._raw_img1 = cv2.imread(image_one_file_path, cv2.IMREAD_COLOR)
        self._raw_img2 = cv2.imread(image_two_file_path, cv2.IMREAD_COLOR)

        self._hsv_img1 = self.convert_hsv_and_remove_luminance(self._raw_img1)
        self._hsv_img2 = self.convert_hsv_and_remove_luminance(self._raw_img2)

        self._calibration_lookup = self.load_calibration_data(calib_data_file_path)
        self._calib_data_file_path = calib_data_file_path

        # self.setup_image_gui(self._hsv_img1)
        # self.setup_image_gui(self._raw_img2)
        #
        # print TSEUtils.get_smallest_key_value_dict(self._lookup_table)
        #
        # plt.show()

    def load_calibration_data(self, file_path):
        raw_data = TSEFileIO.read_file(file_path, split_delimiter=",", start_position=1)
        return dict(TSEUtils.string_list_to_int_list(raw_data))

    def setup_image_gui(self, image):
        plt.figure()
        plt.axis("off")
        canvas = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # canvas.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        # canvas.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def convert_hsv_and_remove_luminance(self, source_image):
        hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)

        # Set the 'V' channel of each pixel in the image to '0' (i.e remove it)
        hsv_image[:, :, 2] = 0

        return hsv_image

    def search_image(self, patch_height):

        smallest_key = TSEUtils.get_smallest_key_dict(self._calibration_lookup)

        image_height, image_width = self._hsv_img2.shape[:2]

        image_centre_x = math.floor(image_width / 2)

        for i in range(smallest_key, image_height - patch_height):

            calibrated_patch_width = self._calibration_lookup[i]
            patch_half_width = math.floor(calibrated_patch_width / 2)

            patch_origin_x = (image_centre_x - patch_half_width)
            patch_end_x = (image_centre_x + patch_half_width)

            roi = self._hsv_img2[i:i + patch_height, patch_origin_x:patch_end_x]

            cv2.imshow("ROI", roi)
            cv2.waitKey(100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("calib_data_file", help="the file containing the calibration data")
    parser.add_argument("input_image_1", help="the first image")
    parser.add_argument("input_image_2", help="the second image")

    args = parser.parse_args()

    match = TemplateMatching(args.input_image_1, args.input_image_2, args.calib_data_file)

    match.search_image(40)

if __name__ == '__main__':  # if the function is the main function ...
    main()
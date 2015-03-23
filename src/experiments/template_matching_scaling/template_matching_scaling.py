from __future__ import division

import argparse

from tsefileio import TSEFileIO
from tseutils import TSEUtils

__author__ = 'connorgoddard'


class TemplateMatching:
    def __init__(self, file_path):
        self._lookup_table = self.load_calibration_data(file_path)
        self._calib_data_file_path = file_path
        print self._lookup_table
        print self._calib_data_file_path

    def load_calibration_data(self, file_path):
        raw_data = TSEFileIO.read_file(file_path, split_delimiter=",", start_position=1)
        return TSEUtils.string_list_to_int_list(raw_data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("calib_data_file", help="the file containing the calibration data")
    parser.add_argument("input_image_1", help="the first image")
    parser.add_argument("input_image_2", help="the second image")

    args = parser.parse_args()

    foo = TemplateMatching(args.calib_data_file)


if __name__ == '__main__':  # if the function is the main function ...
    main()
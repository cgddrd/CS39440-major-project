from __future__ import division

import cv2
import argparse
import matplotlib.pyplot as plt

from point import Point
from fileio import FileIO

__author__ = 'connorgoddard'

def load_calibration_data():

    raw_data = FileIO.read_file("../perspective_calibration/data/calibdata_23_03_15_11_07_04.txt", split_delimiter=",", start_position=1)

    # We remove the first element in the returned array, as this is the
    # raw_data.pop(0)

    for val in raw_data:
        print val

    # print dict(raw_data)


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("input_image_1", help="the first image")
    # parser.add_argument("input_image_2", help="the second image")
    # parser.add_argument("calib_data_file", help="the file containing the calibration data")
    #
    # args = parser.parse_args()

    # filename = "calibdata_{0}.txt".format(datetime.datetime.utcnow().strftime("%d_%m_%y_%H_%M_%S"))

    load_calibration_data()

if __name__ == '__main__':  # if the function is the main function ...
    main()
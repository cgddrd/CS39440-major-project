from __future__ import division

import cv2
import numpy as np
import argparse
import os

__author__ = 'connorgoddard'

def RectColourArgument(raw_argument_string):

    try:
        red, green, blue = raw_argument_string.split(',')

        # Strip out any whitespace either side of arguments.
        return int(red.strip()), int(green.strip()), int(blue.strip())
    except:
        raise argparse.ArgumentTypeError("Colour format expected \"<red>, <green>, <blue>\".")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--image-height', dest="image_height", type=int, default=480)
    parser.add_argument('--image-width', dest="image_width", type=int, default=640)

    parser.add_argument('--output-file-prefix', dest="file_prefix", type=str, required=True)
    parser.add_argument('--output-folder-path', dest="folder_path", type=str, default="./output/")

    parser.add_argument('--rect-height', dest="rect_height", type=int, required=True)
    parser.add_argument('--rect-width', dest="rect_width", type=int, required=True)

    parser.add_argument('--bg-colour', dest="bg_colour", type=RectColourArgument, required=True)

    parser.add_argument('--rect-colour', dest="rect_colour", type=RectColourArgument, required=True)

    parser.add_argument('--rect-y-origin', dest="rect_y_origin", type=int, required=True)
    parser.add_argument('--rect-x-origin', dest="rect_x_origin", type=int, default=-1)

    parser.add_argument('--scaled-rect-y-origin', dest="scaled_rect_y_origin", type=int, required=True)
    parser.add_argument('--scaled-rect-x-origin', dest="scaled_rect_x_origin", type=int, default=-1)

    parser.add_argument('--scale-factor', dest="scale_factor", type=float, required=True)

    args = vars(parser.parse_args())

    small_rec_width = int(args['rect_width'])
    small_rec_height = int(args['rect_height'])

    large_rec_width = int(round(small_rec_width * args['scale_factor']))
    large_rec_height = int(round(small_rec_height * args['scale_factor']))

    img_small_rec = np.full((args['image_height'], args['image_width'], 3), args['bg_colour'], np.uint8)
    img_large_rec = np.full((args['image_height'], args['image_width'], 3), args['bg_colour'], np.uint8)
    img_small_large_rec = np.full((args['image_height'], args['image_width'], 3), args['bg_colour'], np.uint8)

    image_centre = int(round(img_small_rec.shape[1] / 2.0))

    small_rec_x_origin = int(image_centre - int(round(small_rec_width / 2.0))) if args['rect_x_origin'] == -1 else int(args['rect_x_origin'])

    large_rec_x_origin = int(image_centre - int(round(large_rec_width / 2.0))) if args['scaled_rect_x_origin'] == -1 else int(args['scaled_rect_x_origin'])

    cv2.rectangle(img_small_rec, (small_rec_x_origin, int(args['rect_y_origin'])), (small_rec_x_origin + small_rec_width, int(args['rect_y_origin']) + small_rec_height), (args['rect_colour']), -1)
    cv2.rectangle(img_large_rec, (large_rec_x_origin, int(args['scaled_rect_y_origin'])), (large_rec_x_origin + large_rec_width, int(args['scaled_rect_y_origin']) + large_rec_height), (args['rect_colour']), -1)

    cv2.rectangle(img_small_large_rec, (small_rec_x_origin, int(args['rect_y_origin'])), (small_rec_x_origin + small_rec_width, int(args['rect_y_origin']) + small_rec_height), (args['rect_colour']), -1)
    cv2.rectangle(img_small_large_rec, (large_rec_x_origin, int(args['scaled_rect_y_origin'])), (large_rec_x_origin + large_rec_width, int(args['scaled_rect_y_origin']) + large_rec_height), (args['rect_colour']), -1)

    cv2.imwrite(os.path.join(args['folder_path'], "{0}_original.jpg".format(args['file_prefix'])), img_small_rec)
    cv2.imwrite(os.path.join(args['folder_path'], "{0}_scaled.jpg".format(args['file_prefix'])), img_large_rec)
    cv2.imwrite(os.path.join(args['folder_path'], "{0}_calib.jpg".format(args['file_prefix'])), img_small_large_rec)

if __name__ == "__main__":
    main()
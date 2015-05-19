from __future__ import division
__author__ = 'connorgoddard'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import os


def run_sim(folder_path, images, patch_height, patch_width, start_y):

    image1 = cv2.imread(os.path.join(folder_path, images[0]))
    image2 = cv2.imread(os.path.join(folder_path, images[1]))

    data1 = []
    data2 = []

    image_height, image_width = image1.shape[:2]

    image_width_centre = int(image_width / 2)
    half_patch_width = int(patch_width / 2)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    for i in xrange(start_y, (image_height - patch_height)):

        roi_image1 = image1_gray[i:i+patch_height, image_width_centre - half_patch_width: image_width_centre + half_patch_width]
        roi_image1_edges = cv2.Canny(roi_image1, 150, 250)
        # roi_image1_contours, hierarchy = cv2.findContours(roi_image1_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        roi_image2 = image2_gray[i:image_height, image_width_centre - half_patch_width: image_width_centre + half_patch_width]
        roi_image2_edges = cv2.Canny(roi_image2, 150, 250)

        roi_image2_edges_height = roi_image2_edges.shape[0]

        score = -1
        best_y = i

        stop = False

        res = cv2.matchTemplate(roi_image2_edges, roi_image1_edges, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        cv2.imshow("Original - IMG1", roi_image1)
        cv2.imshow("Original - IMG2", roi_image2)
        cv2.imshow("Edges - IMG1", roi_image1_edges)
        cv2.imshow("Edges - IMG2", roi_image2_edges)

        cv2.waitKey(1000)

        data1.append(i)
        data2.append(max_loc[1])

        cv2.rectangle(image2, ((image_width_centre - half_patch_width), i), ((image_width_centre + half_patch_width), (start_y + patch_height)), (255, 0, 0), 3)
        cv2.rectangle(image2, ((image_width_centre - half_patch_width), i + best_y), ((image_width_centre + half_patch_width), (i + best_y + patch_height)), (0, 255, 0), 3)

    cv2.imshow("contours", image2)

    plt.plot(np.array(data1), np.array(data2), 'g-')
    plt.show()


def main():

    # images2 = ["../eval_data/motion_images/flat_10cm/IMG1.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG2.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG3.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG4.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG5.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG6.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG7.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG8.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG9.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG10.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG11.JPG",
    #           "../eval_data/motion_images/flat_10cm/IMG12.JPG"]
    #
    # images = ["../eval_data/motion_images/wiltshire_outside_10cm/IMG1.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG2.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG3.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG4.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG5.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG6.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG7.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG8.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG9.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG10.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG11.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG12.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG13.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG14.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG15.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG16.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG17.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG18.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG19.JPG",
    #           "../eval_data/motion_images/wiltshire_outside_10cm/IMG20.JPG"]

    parser = argparse.ArgumentParser()

    # "nargs='+'" tells 'argparse' to allow for a list of values to be accepted within this parameter.
    parser.add_argument('--source-folder', help='Source folder from which to load images', dest="source_folder", type=str, required=True)
    parser.add_argument('--images', help='List of image file names to load from inside the source-folder. Please note: The order of image names in this list determine the order of images processed by the application.', nargs='+', dest="images", type=str, required=True)

    parser.add_argument('--patch-height', help='Height of the template patch in pixels.', dest="patch_height", type=int, required=True)
    parser.add_argument('--patch-width', help='Width of the template patch in pixels.', dest="patch_width", type=int, required=True)
    parser.add_argument('--y-origin', help='Y origin to begin the template search from.', dest="y_start", type=int, default=0)

    args = vars(parser.parse_args())

    run_sim(args['source_folder'], args['images'], args['patch_height'], args['patch_width'], args['y_start'])

    k = cv2.waitKey(0) & 0xFF

    if k == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
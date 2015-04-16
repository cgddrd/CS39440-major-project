from __future__ import division
__author__ = 'connorgoddard'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def run_sim(images, patch_height, patch_width, start_y):

    image1 = cv2.imread(images[0])
    image2 = cv2.imread(images[1])

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

        # cv2.imshow("res", res)
        # cv2.waitKey(10)

        # for j in xrange(0, (roi_image2_edges_height - patch_height)):
        #
        #     search_window = roi_image2_edges[j:j+patch_height, 0:patch_width]
        #
        #     # search_window_contours, hierarchy = cv2.findContours(search_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        #     # cv2.imshow("image1_roi", roi_image1_edges)
        #     # cv2.imshow("image2_roi", roi_image2_edges)
        #     # cv2.imshow("search_window", search_window)
        #     #
        #     # cv2.waitKey(10)
        #
        #     current_score = 0
        #
        #     # for l in range(len(roi_image1_contours)):
        #     #     for k in range(len(search_window_contours)):
        #     #         current_score += cv2.matchShapes(search_window_contours[k],roi_image1_contours[l],cv2.cv.CV_CONTOURS_MATCH_I1, 0.0)
        #
        #     # current_score = cv2.norm(roi_image1_edges, search_window, cv2.NORM_L2)
        #
        #     # Apply template Matching
        #     res = cv2.matchTemplate(search_window, roi_image1_edges, cv2.TM_CCORR_NORMED)
        #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #
        #     current_score = max_val
        #
        #     if (score == -1) or (current_score > score):
        #         score = current_score
        #         best_y = j
        #
        #     elif current_score < score:
        #
        #         stop = True
        #
        #     # print score
        #
        #     if stop:
        #         break

        # if (score == -1) or (current_score > score):
        #     score = current_score
        #     best_y = j

        data1.append(i)
        data2.append(max_loc[1])

        # print max_loc

        cv2.rectangle(image2, ((image_width_centre - half_patch_width), i), ((image_width_centre + half_patch_width), (start_y + patch_height)), (255, 0, 0), 3)
        cv2.rectangle(image2, ((image_width_centre - half_patch_width), i + best_y), ((image_width_centre + half_patch_width), (i + best_y + patch_height)), (0, 255, 0), 3)

    cv2.imshow("contours", image2)

    plt.plot(np.array(data1), np.array(data2), 'g-')
    plt.show()


def main():

    images2 = ["../eval_data/motion_images/flat_10cm/IMG1.JPG",
              "../eval_data/motion_images/flat_10cm/IMG2.JPG",
              "../eval_data/motion_images/flat_10cm/IMG3.JPG",
              "../eval_data/motion_images/flat_10cm/IMG4.JPG",
              "../eval_data/motion_images/flat_10cm/IMG5.JPG",
              "../eval_data/motion_images/flat_10cm/IMG6.JPG",
              "../eval_data/motion_images/flat_10cm/IMG7.JPG",
              "../eval_data/motion_images/flat_10cm/IMG8.JPG",
              "../eval_data/motion_images/flat_10cm/IMG9.JPG",
              "../eval_data/motion_images/flat_10cm/IMG10.JPG",
              "../eval_data/motion_images/flat_10cm/IMG11.JPG",
              "../eval_data/motion_images/flat_10cm/IMG12.JPG"]

    images = ["../eval_data/motion_images/wiltshire_outside_10cm/IMG1.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG2.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG3.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG4.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG5.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG6.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG7.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG8.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG9.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG10.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG11.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG12.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG13.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG14.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG15.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG16.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG17.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG18.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG19.JPG",
              "../eval_data/motion_images/wiltshire_outside_10cm/IMG20.JPG"]

    run_sim(images, 100, 200, 0)

    cv2.waitKey()



if __name__ == "__main__":
    main()
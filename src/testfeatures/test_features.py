from __future__ import division
import cv2

import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

from circular_buffer import CircularBuffer


class TrackedFeature(CircularBuffer):

    _history_threshold = 7
    _is_smooth = False
    _smooth_transition_limit = 0.25
    _smooth_rotation_limit = math.radians(30)
    _smooth_score = 0
    _smooth_score_change = 0

    def __init__(self):
        CircularBuffer.__init__(self, self._history_threshold)

    def add(self, new_value):

        super(TrackedFeature, self).add(new_value)

        if self.is_full():
            self.grade_smoothness()

    def grade_smoothness(self):

        current_point = self.get_history_value(0)
        prev_point = self.get_history_value(-1)
        second_point = self.get_history_value(-2)
        sixth_point = self.get_history_value(-6)

        # One back in history
        dx1 = current_point[0] - prev_point[0]
        dy1 = current_point[1] - prev_point[1]
        direction1 = math.atan2(dy1, dx1)

        # Two back in history
        dx2 = current_point[0] - second_point[0]
        dy2 = current_point[1] - second_point[1]
        direction2 = math.atan2(dy2, dx2)

        # Six back in history
        dx6 = current_point[0] - sixth_point[0]
        dy6 = current_point[1] - sixth_point[1]
        direction6 = math.atan2(dy6, dx6)

        # simple distance (approximation of real Euclidean distance)
        distance2 = abs(dx2) + abs(dy2)
        distance1 = abs(dx1) + abs(dy1)

        direction_change_6_to_2 = abs(self.subtract_angles(direction6, direction2))
        direction_change_2_to_1 = abs(self.subtract_angles(direction2, direction1))

        prev_change_is_smooth = (distance2 < self._smooth_transition_limit) or (direction_change_6_to_2 < self._smooth_rotation_limit)
        current_change_is_smooth = (distance1 < self._smooth_transition_limit) or (direction_change_2_to_1 < self._smooth_rotation_limit)

        if prev_change_is_smooth and current_change_is_smooth:

            self._is_smooth = True
            self._smooth_score_change = -1

        else:

            self._is_smooth = False

            if (current_change_is_smooth is True) and (prev_change_is_smooth is False):

                self._smooth_score_change = 5

            else:

                self._smooth_score_change = 8

    @staticmethod
    def subtract_angles(angle_1, angle_2):

        delta = angle_1 - angle_2

        while delta >= math.pi:
            delta -= math.pi

        while delta < math.pi:
            delta += math.pi

        return delta

    def apply_score_change(self):

        self._smooth_score += self._smooth_score_change

    def is_out(self):

        return self._smooth_score > 10

def calc_moving_average_array(values, window, mode='valid'):

    weights = np.repeat(1.0, window)/window

    # Including 'valid' MODE will REQUIRE there to be enough data points.
    return np.convolve(values, weights, mode)

# Modified from original source: http://stackoverflow.com/a/16562028
def filter_outliers_ab_dist_median_indices(data, ab_dist_median_factor=2.):

    # Ensure we are dealing with a numpy array before operating.
    data = np.array(data)

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    # 'np.where' returns the indices of the elements that match the mask. See: http://stackoverflow.com/a/9891802
    indices = np.where(s < ab_dist_median_factor)[0]

    return indices

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):

# adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)

    # calc angle of the arrow
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])

    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi / 4)))

    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi / 4)))

    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

def calc_farneback_flow(image1, image2):

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, 0.5, 3, 25, 3, 5, 1.1, 0)

    cv2.imshow('Dense Optical Flow', draw_flow(image2_gray, flow))


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)

    y = y.astype(int)
    x = x.astype(int)

    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:

        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        cv2.circle(vis, (x2, y2), 1, (255, 0, 0), -1)

    return vis

def calc_lk_tracking_flow(image1, image2, feature_params, lk_params, use_subpix=False):

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(image1_gray, **feature_params)

    p0 = np.float32(corners).reshape(-1, 1, 2)

    p1, st, err = cv2.calcOpticalFlowPyrLK(image1_gray, image2_gray, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(image2_gray, image1_gray, p1, None, **lk_params)

    d = abs(p0-p0r).reshape(-1, 2).max(-1)

    good_points = d < 1

    new_pts = []

    image2 = cv2.addWeighted(image2,0.5,image1,0.5,0)

    for pts, is_good_point, corners in itertools.izip(p1, good_points, p0):

        if is_good_point and ((pts[0][1] - corners[0][1] >= 0)):
            new_pts.append([pts[0][0], pts[0][1]])
            cv2.circle(image2, (pts[0][0], pts[0][1]), 5, thickness=2, color=(255,255,0))
            cv2.circle(image2, (corners[0][0], corners[0][1]), 5, thickness=2, color=(255,0,0))
            draw_arrow(image2, (corners[0][0], corners[0][1]), (pts[0][0], pts[0][1]), (255, 255, 255))

    return image2

def calc_lk_patches_flow(image1, image2, feature_params, patch_size, max_val_threshold, use_subpix=False):

    count = 0
    raw_results = {}
    half_patch_size = round(patch_size / 2)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image1_gray_height, image1_gray_width = image1_gray.shape[:2]

    corners = cv2.goodFeaturesToTrack(image1_gray, **feature_params)

    if use_subpix:
        subpix_params = dict(winSize=(10, 10), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.1))
        cv2.cornerSubPix(image1_gray, corners, **subpix_params)

    corners = np.array(corners).astype(int)

    for i in corners:

        sorted_x, sorted_y = i.ravel()

        if ((sorted_y - half_patch_size >= 0) and (sorted_y + half_patch_size <= image1_gray_height)) and ((sorted_x - half_patch_size >= 0) and (sorted_x + half_patch_size <= image1_gray_width)):

            patch = image1_gray[sorted_y - half_patch_size:sorted_y + half_patch_size, sorted_x - half_patch_size:sorted_x + half_patch_size]

            search_window = image2_gray[sorted_y - half_patch_size:image1_gray_height, sorted_x - half_patch_size:sorted_x + half_patch_size]

            res = cv2.matchTemplate(search_window, patch, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= max_val_threshold:

                cv2.rectangle(image1, (int(round(sorted_x - half_patch_size)), int(round(sorted_y - half_patch_size))),
                      (int(round(sorted_x + half_patch_size)), int(round(sorted_y + half_patch_size))), 255, 2)

                cv2.circle(image1, (sorted_x, sorted_y), 3, 255, -1)

                cv2.putText(image1, "{0}".format(count), (sorted_x + 10, sorted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

                top_left = (int(round(sorted_x - half_patch_size + max_loc[0])), int(round(sorted_y - half_patch_size + max_loc[1])))

                located_centre = (int(round(sorted_x + max_loc[0])), int(round(sorted_y + max_loc[1])))

                bottom_right = (top_left[0] + patch_size, top_left[1] + patch_size)

                cv2.rectangle(image2, (int(round(sorted_x - half_patch_size)), int(round(sorted_y - half_patch_size))),
                              (int(round(sorted_x + half_patch_size)), int(round(sorted_y + half_patch_size))), 255, 2)

                cv2.rectangle(image2, top_left, bottom_right, (0, 255, 0), 2)

                draw_arrow(image2, (sorted_x, sorted_y), (int(top_left[0] + half_patch_size), int(top_left[1] + half_patch_size)), (255, 255, 255))

                cv2.putText(image2, "{0}".format(count), (sorted_x, sorted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255))

                cv2.putText(image2, "{0}".format(count), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255))

                if sorted_y in raw_results:
                    raw_results[sorted_y].append((located_centre[1] - sorted_y))
                else:
                    raw_results[sorted_y] = [(located_centre[1] - sorted_y)]

                count += 1

    averaged_results = {}

    for key in raw_results:

        data = np.array(raw_results[key])

        average = np.mean(data)

        averaged_results[key] = average

    sorted_x = []
    sorted_y = []

    for key in sorted(averaged_results):

        sorted_x.append(key)
        sorted_y.append(averaged_results[key])

    filtered_y_indices = filter_outliers_ab_dist_median_indices(sorted_y)

    filtered_x = np.array(sorted_x)[filtered_y_indices]
    filtered_y = np.array(sorted_y)[filtered_y_indices]

    y_moving_average = calc_moving_average_array(np.array(filtered_y), 10)

    plt.plot(filtered_x[len(filtered_x) - len(y_moving_average):], y_moving_average, "b-")

    plt.plot(filtered_x, filtered_y, "g-")


    return image2

# Modified from original source: http://stackoverflow.com/a/17385776/4768230
def cut_array2d(array, shape):
    arr_shape = np.shape(array)
    xcut = np.linspace(0,arr_shape[0],shape[0]+1).astype(np.int)
    ycut = np.linspace(0,arr_shape[1],shape[1]+1).astype(np.int)
    blocks = []
    xextent = []
    yextent = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            blocks.append(array[xcut[i]:xcut[i+1],ycut[j]:ycut[j+1]])
            xextent.append([xcut[i],xcut[i+1]])
            yextent.append([ycut[j],ycut[j+1]])
    return xextent,yextent,blocks

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle


def filter_direction(old_points, new_points):

    bool_result = np.ones(len(old_points), dtype=bool)

    for i, (new, old) in enumerate(zip(new_points, old_points)):

        a, b = new.ravel()
        c, d = old.ravel()

        bool_result[i] = (b - d >= 0)

    return bool_result


def filter_angle(old_points, new_points, vectors):

    bool_result = np.ones(len(old_points), dtype=bool)

    for i, (new, old) in enumerate(zip(new_points, old_points)):

        a, b = new.ravel()
        c, d = old.ravel()

        new_vector = ((a-c), (d-b))

        if i in vectors:
            old_vector = vectors[i]
            angle = angle_between(new_vector, old_vector)
            bool_result[i] = (math.degrees(angle) > 30)
            print "done something"

        vectors[i] = new_vector

    return bool_result


def run_sim(images, feature_params, lk_params, subpix_params):

    count = 0

    # Create some random colors
    color = np.random.randint(0,255,(1000,3))

    old_gray = None
    old_points = None
    mask = None
    feature_history = {}

    for image in images:

        image_loaded = cv2.imread(image)

        image_height, image_width = image_loaded.shape[:2]
        image_height_quarter = int(round(int(image_loaded.shape[0]) / 4))

        if count == 0:

            mask = np.zeros_like(image_loaded)

            old_gray = cv2.cvtColor(image_loaded[image_height_quarter: image_height, 0:image_width], cv2.COLOR_BGR2GRAY)
            old_points = cv2.goodFeaturesToTrack(old_gray, **feature_params)

            cv2.cornerSubPix(old_gray, old_points, **subpix_params)

        else:

            new_gray = cv2.cvtColor(image_loaded[image_height_quarter: image_height, 0:image_width], cv2.COLOR_BGR2GRAY)

            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)

            # new_points_two, st, err = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, new_points, None, **lk_params)
            # d = abs(old_points-new_points_two).reshape(-1, 2).max(-1)
            # ok_points = d < 1

            good_new = new_points[st==1]
            good_old = old_points[st==1]

            # Filter "good" points on the Y direction
            dir_filter = filter_direction(good_old, good_new)

            good_new = good_new[dir_filter]
            good_old = good_old[dir_filter]

            angle_filter = filter_direction(good_old, good_new)

            good_new = good_new[angle_filter]
            good_old = good_old[angle_filter]

            # mask = np.zeros_like(image_loaded)

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                cv2.circle(image_loaded, (a, int(image_height_quarter + b)), 5, (0, 255, 0), -1)
                cv2.circle(image_loaded, (c, int(image_height_quarter + d)), 5, (0, 0, 255), -1)
                cv2.line(mask, (a, int(image_height_quarter + b)), (c, int(image_height_quarter + d)), color[i].tolist(), 2)

            img = cv2.add(image_loaded, mask)

            # Now update the previous frame and previous points
            old_gray = new_gray.copy()

            if len(good_new.reshape(-1,1,2)) <= 25:
                old_points = cv2.goodFeaturesToTrack(old_gray, **feature_params)
                cv2.cornerSubPix(old_gray, old_points, **subpix_params)
                print "Too little points, need to find some more."
            else:
                old_points = good_new.reshape(-1,1,2)

            cv2.imshow('frame', img)
            cv2.waitKey(2000)

        count += 1



__author__ = 'connorgoddard'

def main():

    image1_wiltshire_outside = cv2.imread("../eval_data/motion_images/wiltshire_outside_10cm/IMG1.JPG")
    image2_wiltshire_outside = cv2.imread("../eval_data/motion_images/wiltshire_outside_10cm/IMG2.JPG")

    image1_wiltshire_inside = cv2.imread("../eval_data/motion_images/wiltshire_inside_15cm/IMG1.JPG")
    image2_wiltshire_inside = cv2.imread("../eval_data/motion_images/wiltshire_inside_15cm/IMG2.JPG")

    image1_flat = cv2.imread("../eval_data/motion_images/flat_10cm/IMG1.JPG")
    image2_flat = cv2.imread("../eval_data/motion_images/flat_10cm/IMG2.JPG")

    # calc_farneback_flow(image1, image2)

    feature_params = dict(maxCorners=1000, qualityLevel=0.005, minDistance=20)

    lk_params = dict(winSize=(40, 40), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    subpix_params = dict(winSize=(20, 20), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.1))

    # cv2.imshow("Flat", calc_lk_tracking_flow(image1_flat, image2_flat, feature_params, lk_params))
    # cv2.imshow("Outside", calc_lk_tracking_flow(image1_wiltshire_outside, image2_wiltshire_outside, feature_params, lk_params))
    # cv2.imshow("Inside", calc_lk_tracking_flow(image1_wiltshire_inside, image2_wiltshire_inside, feature_params, lk_params))
    #
    # cv2.imshow("Inside_Patches", calc_lk_patches_flow(image1_flat, image2_flat, feature_params, 100, 0.5))
    #
    # cv2.imshow("Outside_Patches", calc_lk_patches_flow(image1_wiltshire_outside, image2_wiltshire_outside, dict(maxCorners=1000, qualityLevel=0.005, minDistance=20), 100, 0.2))
    #
    # plt.show()

    # x,y,split = cut_array2d(image1_flat, (2, 1))
    #
    # cv2.imshow("sddas", split[0])
    #
    # cv2.imshow("sddas2", split[1])
    #
    # cv2.imshow("image1", image1_flat)

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

    run_sim(images, feature_params, lk_params, subpix_params)

    cv2.waitKey()

if __name__ == "__main__":
    main()
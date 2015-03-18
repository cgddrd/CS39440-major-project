from __future__ import division

__author__ = 'connorgoddard'

import math

import cv2
import argparse
import matplotlib.pyplot as plt

line_coords = []

count = 0
slope = 0


def calc_point(start_point, end_point, start_point2, end_point2):
    global slope

    height, width, depth = img1.shape

    i = start_point[1]

    while i < height:

        new_y = i

        # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
        if (end_point[0] - start_point[0]) != 0:

            slope = float((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
            new_x = (((new_y - start_point[1]) / slope) + start_point[0])

            slope2 = float((end_point2[1] - start_point2[1]) / (end_point2[0] - start_point2[0]))
            new_x2 = (((new_y - start_point2[1]) / slope2) + start_point2[0])

        else:

            new_x = start_point[0]
            new_x2 = start_point2[0]

        cv2.circle(img1, (int(new_x), new_y), 2, (0, 0, 255), -1)

        cv2.circle(img1, (int(new_x2), new_y), 2, (0, 0, 255), -1)

        cv2.line(img1, (int(new_x), new_y), (int(new_x2), new_y), (0, 255, 0), 1)

        i += 1


def calc_point_reflection(start_point, end_point):
    global slope

    height, width, depth = img1.shape

    half_width = width / 2

    start_point2 = (half_width + (half_width - start_point[0]), start_point[1])
    end_point2 = (half_width + (half_width - end_point[0]), end_point[1])

    i = start_point[1]

    while i < height:

        new_y = i

        # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
        if (end_point[0] - start_point[0]) != 0:

            slope = float((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
            new_x = (((new_y - start_point[1]) / slope) + start_point[0])

            slope2 = float((end_point2[1] - start_point2[1]) / (end_point2[0] - start_point2[0]))
            new_x2 = (((new_y - start_point2[1]) / slope2) + start_point2[0])

        else:

            new_x = start_point[0]
            new_x2 = start_point2[0]

        cv2.circle(img1, (int(new_x), new_y), 2, (0, 0, 255), -1)

        cv2.circle(img1, (int(new_x2), new_y), 2, (0, 0, 255), -1)

        cv2.line(img1, (int(new_x), new_y), (int(new_x2), new_y), (0, 255, 0), 1)

        i += 1


def calc_vec_magnitude(point_1, point_2):
    vx = (point_2[0] - point_1[0])
    vy = (point_2[1] - point_1[1])

    return math.sqrt((vx * vx) + (vy * vy))


def add_new_point(mouse_x, mouse_y):
    line_coords.append((mouse_x, mouse_y))


def handle_mouse(event, x, y, flags, other):
    global count

    if event == cv2.EVENT_LBUTTONDOWN:

        if count < 4:

            add_new_point(x, y)
            cv2.circle(img1, (x, y), 2, (255, 0, 0), -1)

            # If we have an even number of points, then we can draw a line between the last two points added to the collection.
            if len(line_coords) % 2 == 0:
                cv2.line(img1, (line_coords[len(line_coords) - 1][0], line_coords[len(line_coords) - 1][1]),
                         (line_coords[len(line_coords) - 2][0], line_coords[len(line_coords) - 2][1]), (0, 255, 0), 1)

            count += 1


parser = argparse.ArgumentParser()
parser.add_argument("inputImage", help="the first image")
args = parser.parse_args()

img1 = cv2.imread(args.inputImage, cv2.IMREAD_COLOR)


def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(int(event.xdata), int(event.ydata))


def onpress(event):
    if event.key == 'q':
        img2 = cv2.imread("../eval_data/motion_images/flat_10cm/IMG12.JPG", cv2.IMREAD_COLOR)
        plt.clf()
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.draw()
        # plt.show()


plt.axis("off")

implot = plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
cis = implot.figure.canvas.mpl_connect('button_press_event', onclick)
cikey = implot.figure.canvas.mpl_connect('key_press_event', onpress)

plt.show()
from __future__ import division

__author__ = 'connorgoddard'

import math

import cv2
import argparse

lineCoords = []

count = 0

slope = 0


def calc_point(start_point, end_point):

    global slope

    vx = (end_point[0] - start_point[0])
    vy = (end_point[1] - start_point[1])

    magnitude = math.sqrt((vx * vx) + (vy * vy))

    vx /= magnitude
    vy /= magnitude

    # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
    if vx == 0:
        slope = 0
    else:
        slope = vy / vx

    # We add the magnitude if we want to draw a point on the line that EXTENDS PAST THE TWO ORIGINAL POINTS.
    # px = int(float(start_point[0] + vx * (magnitude + 20)))
    # py = int(float(start_point[1] + vy * (magnitude + 20)))

    new_y = 100 + start_point[1]

    top = 100

    r = int(top / slope)

    r += start_point[0]

    # print slope
    # print top
    # print r

    px = int(float(start_point[0] + (vx * 100)))
    py = int(float(start_point[1] + (vy * 100)))


    print "Start Point XY: {0}, {1}".format(start_point[0], start_point[1])
    print "End Point XY: {0}, {1}\n------------".format(end_point[0], end_point[1])
    print "Slope XY: {0}, {1}".format(r, new_y)
    print "Mag XY: {0}, {1}\n\n----------\n\n".format(px, py)

    cv2.circle(img1, (r, new_y), 2, (0, 0, 255), -1)
    cv2.circle(img1, (start_point[0], start_point[1]), 2, (255, 0, 0), -1)
    cv2.circle(img1, (px, py), 2, (255, 0, 255), -1)

    # Python "for loop": 0 = start, int(magnitude) = end, 5 = increment

    # for i in xrange(0, int(magnitude), 5):
    #
    #     # As we want to draw a point on the line BETWEEN the two points, we simply don't add the magnitude of the line between the points onto the coordinate position.
    #     px = int(float(start_point[0] + (vx * i)))
    #     py = int(float(start_point[1] + (vy * i)))
    #
    #     print px
    #     print py
    #
    #     cv2.circle(img1, (px, py), 2, (0, 0, 255), -1)


def get_distance(point1, point2):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


def in_between(point1, point2, targetpoint):
    # crossproduct = (targetpoint[1] - point1[1]) * (point2[0] - point1[0]) - (targetpoint[0] - point1[0]) * (point2[1] - point1[1])
    #
    # epsilon = sys.float_info.epsilon
    #
    # print abs(crossproduct)
    # print epsilon
    #
    # if abs(crossproduct) <= epsilon:
    # print "point matches"

    print point1
    print point2
    print targetpoint

    print "\n--------\n"

    dist1 = get_distance(point1, targetpoint)
    dist2 = get_distance(targetpoint, point2)
    disttotal = get_distance(point1, point2)

    distcalc = dist1 + dist2

    print distcalc
    print disttotal

    print distcalc - disttotal

    if distcalc == disttotal:
        print "point matches"


def draw():
    global count
    if count >= 2:
        cv2.line(img1, lineCoords[0], lineCoords[1], (0, 255, 0), 1)


def draw_circle(event, x, y, flags, param):

    global count

    if event == cv2.EVENT_LBUTTONDOWN:

        if count < 2:

            cv2.circle(img1, (x, y), 2, (255, 0, 0), -1)
            lineCoords.append((x, y))
            count += 1
            draw()

    if event == cv2.EVENT_RBUTTONDOWN:
        # cv2.circle(img1, (x, y), 2, (0, 0, 255), -1)
        # in_between(lineCoords[0], lineCoords[1], (x, y))

        if count >= 2:

            calc_point(lineCoords[0], lineCoords[1])

            count = 0

            # Empty the list of coordinates.
            lineCoords[:] = []


parser = argparse.ArgumentParser()
parser.add_argument("inputImage", help="the first image")
args = parser.parse_args()

img1 = cv2.imread(args.inputImage, cv2.IMREAD_COLOR)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)


while 1:
    cv2.imshow('image', img1)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
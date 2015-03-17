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

    height, width, depth = img1.shape

    halfwidth = width / 2;

    start_point2 = (halfwidth + (halfwidth - start_point[0]), start_point[1])
    end_point2 = (halfwidth + (halfwidth - end_point[0]), end_point[1])

    print start_point
    print start_point2

    vx = (end_point[0] - start_point[0])
    vy = (end_point[1] - start_point[1])

    vx2 = (end_point2[0] - start_point2[0])
    vy2 = (end_point2[1] - start_point2[1])

    # magnitude = math.sqrt((vx * vx) + (vy * vy))
    #
    # vx /= magnitude
    # vy /= magnitude
    #
    # px = int(float(start_point[0] + (vx * 100)))
    # py = int(float(start_point[1] + (vy * 100)))

    # print "Start Point XY: {0}, {1}".format(start_point[0], start_point[1])
    # print "End Point XY: {0}, {1}\n------------".format(end_point[0], end_point[1])
    # print "Slope XY: {0}, {1}".format(new_x, new_y)

    # Python "for loop": 0 = start, int(magnitude) = end, 5 = increment

    # for i in xrange(0, int(magnitude), 5):
    #
    # # As we want to draw a point on the line BETWEEN the two points, we simply don't add the magnitude of the line between the points onto the coordinate position.
    #     px = int(float(start_point[0] + (vx * i)))
    #     py = int(float(start_point[1] + (vy * i)))
    #
    #     print px
    #     print py
    #
    #     cv2.circle(img1, (px, py), 2, (0, 0, 255), -1)

    i = start_point[1]



    print height, width, depth

    while i < height:

        new_y = i

        # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
        if vx != 0:
            slope = float((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
            new_x = (((new_y - start_point[1]) / slope) + start_point[0])

            slope2 = float((end_point2[1] - start_point2[1]) / (end_point2[0] - start_point2[0]))
            new_x2 = (((new_y - start_point2[1]) / slope2) + start_point2[0])


        else:
            new_x = start_point[0]

        cv2.circle(img1, (int(new_x), new_y), 2, (0, 0, 255), -1)

        cv2.circle(img1, (int(new_x2), new_y), 2, (0, 0, 255), -1)

        cv2.line(img1, (int(new_x), new_y), (int(new_x2), new_y), (0, 255, 0), 1)

        i+=1


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
        count = 0


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

        # if count >= 2:
        #     calc_point(lineCoords[0], lineCoords[1], lineCoords[2], lineCoords[3])

            # count = 0

            calc_point(lineCoords[0], lineCoords[1])

            # Empty the list of coordinates.
            # lineCoords[:] = []


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
from __future__ import division
import math

__author__ = 'connorgoddard'


def calc_vec_magnitude(point_1, point_2):
    vx = (point_2[0] - point_1[0])
    vy = (point_2[1] - point_1[1])

    return math.sqrt((vx * vx) + (vy * vy))


def calc_line_points_reflection(start_point, end_point, image_height, image_width):
    half_width = image_width / 2

    start_point2 = (half_width + (half_width - start_point[0]), start_point[1])
    end_point2 = (half_width + (half_width - end_point[0]), end_point[1])

    i = start_point[1]

    coords1 = []
    coords2 = []

    while i < image_height:

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

        coords1.append((int(new_x), new_y))

        coords2.append((int(new_x2), new_y))

        i += 1

    return coords1, coords2


def calc_line_points(start_point, end_point, start_point2, end_point2, image_height):
    i = start_point[1]

    coords1 = []
    coords2 = []

    while i < image_height:

        new_y = i

        # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
        if (end_point[0] - start_point[0]) != 0 and (end_point2[0] - start_point2[0]) != 0:

            slope = float((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
            new_x = (((new_y - start_point[1]) / slope) + start_point[0])

            slope2 = float((end_point2[1] - start_point2[1]) / (end_point2[0] - start_point2[0]))
            new_x2 = (((new_y - start_point2[1]) / slope2) + start_point2[0])

        else:

            new_x = start_point[0]
            new_x2 = start_point2[0]

        coords1.append((int(new_x), new_y))

        coords2.append((int(new_x2), new_y))

        i += 1

    return coords1, coords2

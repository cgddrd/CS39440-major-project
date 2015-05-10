"""

Module Name: TSEGeometry

Description: Provides mathematical functions relating to the geometric transformation of image pixel coordinates.

"""

from __future__ import division
import math

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


class TSEGeometry:
    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_measure_scale_factor(current_measure, target_measure):
        """
        Calculates the scale factor between a current dimension, and the target dimension. (Assuming both represent the same dimension)

        :param current_measure: The current dimension of the image patch.
        :param target_measure: The target dimension of the image patch.
        :return Float holding the calculated scale factor between the two measures
        """

        return float(target_measure / current_measure)

    @staticmethod
    def scale_coordinate_relative_centre(coordinate, centre_coordinate, scale_factor):
        """
        Applies geometric scaling of the coordinate for an image patch pixel relative to the centre pixel coordinate.

        :param coordinate: The X/Y coordinate for the original pixel.
        :param centre_coordinate: The X/Y coordinate for the centre pixel relative to the original coordinate.
        :param scale_factor: The factor by which to scale the original pixel coordinate by.
        :return Scaled pixel coordinate relative to the original centre coordinate.
        """

        # Calculate the vector between the centre coordinate and the original coordinate.
        vec = (coordinate[0] - centre_coordinate[0], coordinate[1] - centre_coordinate[1])

        # Scale the vector between the centre coordinate and the original coordinate.
        new_vec = ((vec[0] * scale_factor), (vec[1] * scale_factor))

        # Add this scaled vector to the centre coordinate in order to return the final scaled pixel coordinate.
        return int(round(new_vec[0] + centre_coordinate[0])), int((new_vec[1] + centre_coordinate[1]))

    @staticmethod
    def calc_vec_magnitude(point_1, point_2):
        """
        Calculates the magnitude of the vector between two distinct coordinate points.

        :param point_1: The first end point.
        :param point_2: The second end point.
        :return The magnitude of the vector as a Float.
        """

        vx = (point_2[0] - point_1[0])
        vy = (point_2[1] - point_1[1])

        return math.sqrt((vx * vx) + (vy * vy))

    @staticmethod
    def calc_line_points_horizontal_reflection(original_start_point, original_end_point, reflect_axis_x_coord, max_y):
        """
        Calculates the straight-line coordinates between two distinct points following a reflection along the X-axis.

        :param original_start_point: The original start point.
        :param original_end_point: The original end point.
        :param reflect_axis_x_coord: X-axis coordinate upon which the horizontal reflection should fall.
        :param max_y: The maximum Y-coordinate up to which line coordinate points should be calculated (if max_y > original_end_point)
        :return Tuple containing separate X and Y coordinates for all calculated line points.
        """

        # Calculate the reflected start and end coordinates relative to the origin along the X-axis.
        reflected_start_point = (reflect_axis_x_coord + (reflect_axis_x_coord - original_start_point[0]), original_start_point[1])
        reflected_end_point = (reflect_axis_x_coord + (reflect_axis_x_coord - original_end_point[0]), original_end_point[1])

        i = original_start_point[1]

        coords1 = []
        coords2 = []

        # Loop through each Y-coordinate..
        while i <= max_y:

            new_y = i

            # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
            # As the second line will be a complete reflection of the first, we only need to perform a slope check on the first line.
            if (original_end_point[0] - original_start_point[0]) != 0:

                # Calculate the relative X-coordinate along the original line.
                slope = float((original_end_point[1] - original_start_point[1]) / (original_end_point[0] - original_start_point[0]))
                new_x = (((new_y - original_start_point[1]) / slope) + original_start_point[0])

                # Calculate the relative X-coordinate along the reflected line.
                slope2 = float((reflected_end_point[1] - reflected_start_point[1]) / (reflected_end_point[0] - reflected_start_point[0]))
                new_x2 = (((new_y - reflected_start_point[1]) / slope2) + reflected_start_point[0])

            else:

                new_x = original_start_point[0]
                new_x2 = reflected_start_point[0]

            coords1.append((int(round(new_x)), new_y))
            coords2.append((int(round(new_x2)), new_y))

            i += 1

        return coords1, coords2

    @staticmethod
    def calc_line_points(start_point, end_point, start_point2, end_point2, max_y):
        """
        Calculates the straight-line coordinates between two pairs of corresponding points.

        :param start_point: The first start point.
        :param end_point: The first end point.
        :param start_point2: The second start point.
        :param end_point2: The second end point.
        :param max_y: The maximum Y-coordinate up to which line coordinate points should be calculated (if max_y > original_end_point)
        :return Tuple containing separate X and Y coordinates for all calculated line points.
        """

        i = start_point[1]

        coords1 = []
        coords2 = []

        while i <= max_y:

            new_y = i

            # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
            if (end_point[0] - start_point[0]) != 0:

                # Calculate the relative X-coordinate along the original line.
                slope = float((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
                new_x = (((new_y - start_point[1]) / slope) + start_point[0])

            else:
                # If the current slope is zero, then force the new change in X coordinate to zero as well.
                new_x = start_point[0]

            if (end_point2[0] - start_point2[0]) != 0:
                slope2 = float((end_point2[1] - start_point2[1]) / (end_point2[0] - start_point2[0]))
                new_x2 = (((new_y - start_point2[1]) / slope2) + start_point2[0])
            else:
                new_x2 = start_point2[0]

            coords1.append((int(new_x), new_y))
            coords2.append((int(new_x2), new_y))

            i += 1

        return coords1, coords2

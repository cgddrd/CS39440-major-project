from __future__ import division
import math

__author__ = 'connorgoddard'


class TSEGeometry:
    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def calc_measure_scale_factor(current_measure, target_measure):
        return float(target_measure / current_measure)

    @staticmethod
    def scale_coordinate_relative_centre(coordinate, centre_coordinate, scale_factor):

        # Calculate the vector between the centre coordinate and the target coordinate.
        vec = (coordinate[0] - centre_coordinate[0], coordinate[1] - centre_coordinate[1])

        new_vec = ((vec[0] * scale_factor), (vec[1] * scale_factor))

        # Return scaled coordinate relative to the centre of the patch
        return int(new_vec[0] + centre_coordinate[0]), int(new_vec[1] + centre_coordinate[1])

    @staticmethod
    def calc_vec_magnitude(point_1, point_2):
        vx = (point_2[0] - point_1[0])
        vy = (point_2[1] - point_1[1])

        return math.sqrt((vx * vx) + (vy * vy))

    @staticmethod
    def calc_line_points_horizontal_reflection(original_start_point, original_end_point, image_height, image_width, stretch_image_height=False):

        half_width = image_width / 2

        reflected_start_point = (half_width + (half_width - original_start_point[0]), original_start_point[1])
        reflected_end_point = (half_width + (half_width - original_end_point[0]), original_end_point[1])

        i = original_start_point[1]

        coords1 = []
        coords2 = []

        if stretch_image_height is True:

            max_y = image_height

        else:

            max_y = original_end_point[1]

        while i < max_y:

            new_y = i

            # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
            # As the second line will be a complete reflection of the first, we only need to perform a slope check on the first line.
            if (original_end_point[0] - original_start_point[0]) != 0:

                slope = float((original_end_point[1] - original_start_point[1]) / (original_end_point[0] - original_start_point[0]))
                new_x = (((new_y - original_start_point[1]) / slope) + original_start_point[0])

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
    def calc_line_points(start_point, end_point, start_point2, end_point2, image_height):
        i = start_point[1]

        coords1 = []
        coords2 = []

        while i < image_height:

            new_y = i

            # If we happen to have a perfectly vertical line, then we have no slope (prevents division by 0 error)
            if (end_point[0] - start_point[0]) != 0:
                slope = float((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
                new_x = (((new_y - start_point[1]) / slope) + start_point[0])
            else:
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

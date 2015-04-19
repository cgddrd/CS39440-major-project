from __future__ import division

import cv2
import argparse
import matplotlib.pyplot as plt
import datetime

from point import Point
from fileio import FileIO
import geometrymath

__author__ = 'connorgoddard'

class PerspectiveCalibration:

    _calibration_results = []
    _point_coords = []
    _point_count = 0
    _line_slope = 0
    _lock_gui = False

    def __init__(self, image_path, output_file_name_prefix, output_folder_name):

        self._output_file_name_prefix = output_file_name_prefix
        self._file = FileIO(output_folder_name)
        self._original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self._new_img = self._original_img.copy()

        self.setup_image_gui(self._new_img)

    def render_line(self, start_point, end_point, start_point2, end_point2):

        height, width, depth = self._new_img.shape

        # Here we are unwrapping the TUPLE returned from the function into two separate variables (coords_line_1, coords_line_2)
        coords_line_1, coords_line_2 = geometrymath.calc_line_points(start_point, end_point, start_point2, end_point2, height)

        for i in range(len(coords_line_1)):
            cv2.circle(self._new_img, coords_line_1[i], 2, (0, 0, 255), -1)

            cv2.circle(self._new_img, coords_line_2[i], 2, (0, 0, 255), -1)

            cv2.line(self._new_img, coords_line_1[i], coords_line_2[i], (0, 255, 255), 1)

            self._calibration_results.append((coords_line_1[i][1], (coords_line_2[i][0] - coords_line_1[i][0])))

    def render_line_reflection(self, start_point, end_point):

        image_height, image_width = self._new_img.shape[:2]

        # Here we are unwrapping the TUPLE returned from the function into two separate variables (coords_line_1, coords_line_2)
        coords_line_1, coords_line_2 = geometrymath.calc_line_points_reflection(start_point, end_point, image_height, image_width)

        for i in range(len(coords_line_1)):
            cv2.circle(self._new_img, coords_line_1[i], 2, (0, 0, 255), -1)

            cv2.circle(self._new_img, coords_line_2[i], 2, (0, 0, 255), -1)

            cv2.line(self._new_img, coords_line_1[i], coords_line_2[i], (0, 255, 255), 1)

            calibrated_width = (coords_line_2[i][0] - coords_line_1[i][0])

            # Prevent exporting calibration width values > original image width.
            self._calibration_results.append((coords_line_1[i][1], min(image_width, calibrated_width)))

    def add_new_point(self, mouse_x, mouse_y):

        new_point = Point(mouse_x, mouse_y)
        self._point_coords.append(new_point.get_value())

    def on_mouse_click(self, event):

        toolbar = plt.get_current_fig_manager().toolbar

        if toolbar.mode == '' and (event.xdata is not None and event.ydata is not None):

            x = int(event.xdata)
            y = int(event.ydata)

            if self._point_count < 4 and (self._lock_gui is False):

                self.add_new_point(x, y)
                cv2.circle(self._new_img, (x, y), 2, (255, 0, 0), -1)

                # If we have an even number of points, then we can draw a line between the last two points added to the collection.
                if len(self._point_coords) % 2 == 0:
                    cv2.line(self._new_img,
                             (self._point_coords[len(self._point_coords) - 1][0],
                              self._point_coords[len(self._point_coords) - 1][1]),
                             (self._point_coords[len(self._point_coords) - 2][0],
                              self._point_coords[len(self._point_coords) - 2][1]), (0, 255, 0),
                             1)

                self.update_image_gui(self._new_img)
                self._point_count += 1

    def on_key_press(self, event):

        if event.key == 'r':
            self.reset_image_gui()

        elif event.key == 'c':

            if self._lock_gui is False and self._point_count == 4:
                self.render_line(self._point_coords[0], self._point_coords[1], self._point_coords[2],
                                 self._point_coords[3])
                self.update_image_gui(self._new_img)
                self._lock_gui = True

        elif event.key == 'v':

            if self._lock_gui is False and self._point_count == 2:
                self.render_line_reflection(self._point_coords[0], self._point_coords[1])
                self.update_image_gui(self._new_img)
                self._lock_gui = True

        elif event.key == 'w':

            if (self._lock_gui is True) and (len(self._calibration_results) > 0):

                filename = "{0}_{1}.txt".format(self._output_file_name_prefix, datetime.datetime.utcnow().strftime("%d_%m_%y_%H_%M_%S"))

                self._file.write_file(filename, "height,calib_width", self._calibration_results, False)

            else:
                print "Error: Cannot write calibration results to file until results have been generated."

        elif event.key == 'q':
            plt.close()

    def setup_image_gui(self, image):

        plt.axis("off")

        canvas = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        canvas.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        canvas.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show()

    def update_image_gui(self, image):

        plt.clf()
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.draw()

    def reset_image_gui(self):

        self._new_img = self._original_img.copy()
        self.update_image_gui(self._new_img)

        self._point_count = 0

        # Clear collections ready to start again.
        self._point_coords[:] = []
        self._calibration_results[:] = []

        self._lock_gui = False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', dest="input_image", type=str, required=True)
    parser.add_argument('--output-file-prefix', dest="file_prefix", type=str, default="calibdata")
    parser.add_argument('--output-folder-path', dest="folder_path", type=str, default="./output/")

    args = vars(parser.parse_args())

    PerspectiveCalibration(args['input_image'], args['file_prefix'], args['folder_path'])


if __name__ == '__main__':  # if the function is the main function ...
    main()
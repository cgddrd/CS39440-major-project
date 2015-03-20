from __future__ import division
import datetime

import cv2
import argparse
import matplotlib.pyplot as plt

from point import Point
from fileio import FileIO
import geometrymath


__author__ = 'connorgoddard'


class PerspectiveCalibration:
    def __init__(self, image_path, output_filename):
        self._point_coords = []
        self._point_count = 0
        self._line_slope = 0
        self._lock_gui = False
        self._output_filename = output_filename
        self._file = FileIO("./data/")

        self._original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self._new_img = self._original_img.copy()

        self.setup_image_gui(self._new_img)

    def render_line(self, start_point, end_point, start_point2, end_point2):

        height, width, depth = self._new_img.shape

        # Here we are unwrapping the TUPLE returned from the function into two separate variables (coords1, coords2)
        coords1, coords2 = geometrymath.calc_line_points(start_point, end_point, start_point2, end_point2, height)

        file_points = []

        for i in range(len(coords1)):
            cv2.circle(self._new_img, coords1[i], 2, (0, 0, 255), -1)

            cv2.circle(self._new_img, coords2[i], 2, (0, 0, 255), -1)

            cv2.line(self._new_img, coords1[i], coords2[i], (0, 255, 255), 1)

            file_points.append((coords1[i][1], (coords2[i][0] - coords1[i][0])))

        self._file.write_file(self._output_filename, "height,calib_width", file_points, False)

    def render_line_reflection(self, start_point, end_point):

        height, width, depth = self._new_img.shape

        # Here we are unwrapping the TUPLE returned from the function into two separate variables (coords1, coords2)
        coords1, coords2 = geometrymath.calc_line_points_reflection(start_point, end_point, height, width)

        file_points = []

        for i in range(len(coords1)):
            cv2.circle(self._new_img, coords1[i], 2, (0, 0, 255), -1)

            cv2.circle(self._new_img, coords2[i], 2, (0, 0, 255), -1)

            cv2.line(self._new_img, coords1[i], coords2[i], (0, 255, 255), 1)

            file_points.append((coords1[i][1], (coords2[i][0] - coords1[i][0])))

        self._file.write_file(self._output_filename, "height,calib_width", file_points, False)

    def add_new_point(self, mouse_x, mouse_y):
        new_point = Point(mouse_x, mouse_y)
        self._point_coords.append(new_point.get_value())

    def on_mouse_click(self, event):
        toolbar = plt.get_current_fig_manager().toolbar

        if toolbar.mode == '' and (event.xdata != None and event.ydata != None):

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
        self._point_coords[:] = []
        self._lock_gui = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputImage", help="the image")
    args = parser.parse_args()

    filename = "calibdata_{0}.txt".format(datetime.datetime.utcnow().strftime("%d_%m_%y_%H_%M_%S"))

    PerspectiveCalibration(args.inputImage, filename)


if __name__ == '__main__':  # if the function is the main function ...
    main()
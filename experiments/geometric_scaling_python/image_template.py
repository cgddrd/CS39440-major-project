from __future__ import division

__author__ = 'connorgoddard'

import math
import cv2
import argparse
import matplotlib.pyplot as plt

line_coords = []
point_count = 0
slope = 0
lock_gui = False
new_img = 0
original_img = 0


def calc_point(start_point, end_point, start_point2, end_point2):
    global slope

    height, width, depth = new_img.shape

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

        cv2.circle(new_img, (int(new_x), new_y), 2, (0, 0, 255), -1)

        cv2.circle(new_img, (int(new_x2), new_y), 2, (0, 0, 255), -1)

        cv2.line(new_img, (int(new_x), new_y), (int(new_x2), new_y), (255, 255, 0), 1)

        i += 1


def calc_point_reflection(start_point, end_point):
    global slope

    height, width, depth = new_img.shape

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

        cv2.circle(new_img, (int(new_x), new_y), 2, (0, 0, 255), -1)

        cv2.circle(new_img, (int(new_x2), new_y), 2, (0, 0, 255), -1)

        cv2.line(new_img, (int(new_x), new_y), (int(new_x2), new_y), (0, 255, 255), 1)

        i += 1


def calc_vec_magnitude(point_1, point_2):
    vx = (point_2[0] - point_1[0])
    vy = (point_2[1] - point_1[1])

    return math.sqrt((vx * vx) + (vy * vy))


def add_new_point(mouse_x, mouse_y):
    line_coords.append((mouse_x, mouse_y))


def on_mouse_click(event):
    global new_img, point_count

    toolbar = plt.get_current_fig_manager().toolbar

    print toolbar.mode

    if toolbar.mode == '' and (event.xdata != None and event.ydata != None):

        x = int(event.xdata)
        y = int(event.ydata)

        if point_count < 4 and (lock_gui is False):

            add_new_point(x, y)
            cv2.circle(new_img, (x, y), 2, (255, 0, 0), -1)

            # If we have an even number of points, then we can draw a line between the last two points added to the collection.
            if len(line_coords) % 2 == 0:
                cv2.line(new_img, (line_coords[len(line_coords) - 1][0], line_coords[len(line_coords) - 1][1]),
                         (line_coords[len(line_coords) - 2][0], line_coords[len(line_coords) - 2][1]), (0, 255, 0), 1)

            update_image_gui(new_img)
            point_count += 1


def on_key_press(event):
    global lock_gui

    if event.key == 'r':
        reset_image_gui()

    elif event.key == 'c':

        if lock_gui is False and point_count == 4:
            calc_point(line_coords[0], line_coords[1], line_coords[2], line_coords[3])
            update_image_gui(new_img)
            lock_gui = True

    elif event.key == 'v':

        if lock_gui is False and point_count == 2:
            calc_point_reflection(line_coords[0], line_coords[1])
            update_image_gui(new_img)
            lock_gui = True

    elif event.key == 'q':
        plt.close()


def setup_image_gui(image):
    plt.axis("off")

    canvas = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    canvas.figure.canvas.mpl_connect('button_press_event', on_mouse_click)
    canvas.figure.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()


def update_image_gui(image):
    plt.clf()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.draw()


def reset_image_gui():
    global new_img, point_count, lock_gui

    new_img = original_img.copy()
    update_image_gui(new_img)

    point_count = 0
    line_coords[:] = []
    lock_gui = False


def main():
    global original_img, new_img

    parser = argparse.ArgumentParser()
    parser.add_argument("inputImage", help="the first image")
    args = parser.parse_args()

    original_img = cv2.imread(args.inputImage, cv2.IMREAD_COLOR)

    new_img = original_img.copy()

    setup_image_gui(new_img)


if __name__ == '__main__':  # if the function is the main function ...
    main()
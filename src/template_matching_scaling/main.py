from __future__ import division

import cv2
import math
import matplotlib.pyplot as plt
import argparse

from template_matching import TemplateMatching
from tse.tse_matchtype import TSEMatchType
from tse.tse_matchmethod import tse_match_methods

from collections import OrderedDict

__author__ = 'connorgoddard'

def start_tests(image_pairs, patch_sizes, match_methods, config_file, use_scaling=False, scale_centre=True, exhaustive_search=False, use_hsv=True, strip_luminance=True, plot_results=False):

    # Create a dictionary to store the results of all image pairs -> patch sizes -> match methods for a given pair of images.
    image_dict = {}

    # If we are not using the HSV colour space, then prevent the accidental removal of the 3rd channel (Luminance in HSV).
    strip_luminance_sanity = strip_luminance if (use_hsv is True) else False

    if plot_results is False:

        for pair in image_pairs:

            patch_dict = {}

            match = TemplateMatching(pair[0], pair[1], config_file, None, use_hsv, strip_luminance_sanity)

            for patch_size in patch_sizes:

                match_dict = {}

                for match_method in match_methods:
                    match_dict[match_method.match_name] = match.run_template_search(patch_size, match_method, use_scaling, scale_centre, exhaustive_search, plot_results)

                patch_dict[patch_size] = match_dict

            image_dict["{0}_{1}".format(match._image_one_file_name, match._image_two_file_name)] = patch_dict

    else:

        for pair in image_pairs:

            # Create a dictionary to store the results of all patch sizes -> match methods for a given pair of images.
            patch_dict = {}

            plot_count = len(patch_sizes)

            column_max = 2

            row_max = int(math.ceil(plot_count / float(column_max)))

            fig, axes = plt.subplots(row_max, column_max)

            column_count = 0
            row_count = 0

            if row_max > 1:
                match = TemplateMatching(pair[0], pair[1], config_file, axes[row_count, column_count])
            else:
                match = TemplateMatching(pair[0], pair[1], config_file, axes[column_count])

            for patch_size in patch_sizes:

                # Create a dictionary to store the results of all match_methods for a given patch size.
                match_dict = {}

                if row_max > 1:
                    match._plot_axis = axes[row_count, column_count]
                else:
                    match._plot_axis = axes[column_count]

                match._plot_axis.set_xlabel('Row Number (px)')
                match._plot_axis.set_ylabel('Vertical Displacement (px)')
                match._plot_axis.set_title('Patch: {0}px - Images: {1}, {2}'.format(patch_size, match._image_one_file_name, match._image_two_file_name))

                for match_method in match_methods:

                    # Store the results for a given match method.
                    match_dict[match_method.match_name] = match.run_template_search(patch_size, match_method, use_scaling, scale_centre, exhaustive_search, plot_results)

                if column_count == (column_max - 1):
                    row_count += 1
                    column_count = 0
                else:
                    column_count += 1

                patch_dict[patch_size] = match_dict

            # If we do not have an even number of graphs, then we need to remove the last blank one.
            if (plot_count % column_max) != 0:

                if row_max > 1:

                    axes[-1, -1].axis('off')

                else:

                    axes[-1].axis('off')

            image_dict["{0}_{1}".format(match._image_one_file_name, match._image_two_file_name)] = patch_dict

    return image_dict


def InputImagePairArgument(raw_argument_string):

    try:
        x, y = raw_argument_string.split(',')

        # Strip out any whitespace either side of arguments.
        return x.strip(), y.strip()
    except:
        raise argparse.ArgumentTypeError("Image pairs expect format \"<image_1_path>, <image_2_path>\"")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--calibfile', help='Datafile containing the calibration data', dest="calib_file", required=True)
    parser.add_argument('--images', help="Images", dest="image_pairs", type=InputImagePairArgument, nargs='+', required=True)
    parser.add_argument('--patches', nargs='+', dest="patch_sizes", type=int, required=True)
    parser.add_argument('--methods', nargs='+', dest="match_methods", type=str, required=True)
    parser.add_argument('--use-scaling', dest='scaling', action='store_true')
    parser.add_argument('--draw-plot', dest='plot_results', action='store_true')
    parser.add_argument('--exhaustive', dest='exhaustive_search', action='store_true')
    parser.add_argument('--use-rgb', dest='use_rgb', action='store_false')
    parser.add_argument('--strip-luminance', dest='hsv_strip_luminance', action='store_true')
    parser.add_argument('--scale-top', dest='scale_top', action='store_false')

    args = vars(parser.parse_args())

    match_methods = []

    # 'OrderedDict' is used to remove any duplicates.
    for method in list(OrderedDict.fromkeys(args['match_methods'])):

        if method == "DistanceEuclidean":
            match_methods.append(TSEMatchType("DistanceEuclidean", tse_match_methods.DISTANCE_ED, None, "r", reverse_score=True))

        elif method == "DistanceCorr":
            match_methods.append(TSEMatchType("DistanceCorr", tse_match_methods.DISTANCE, cv2.cv.CV_TM_CCORR_NORMED, "b"))

        elif method == "HistCorrel":
            match_methods.append(TSEMatchType("HistCorrel", tse_match_methods.HIST, cv2.cv.CV_COMP_CORREL, "y"))

        elif method == "HistChiSqr":
            match_methods.append(TSEMatchType("HistChiSqr", tse_match_methods.HIST, cv2.cv.CV_COMP_CHISQR, "g", reverse_score=True))

        else:
            parser.error("Error: \"{0}\" is not a valid matching method option.\nSupported Methods: \'DistanceEuclidean\', \'DistanceCorr\', \'HistCorrel\', \'HistChiSqr\' ".format(method))

    # Start the tests using settings passed in as command-line arguments.
    start_tests(args['image_pairs'], list(OrderedDict.fromkeys(args['patch_sizes'])), match_methods, args['calib_file'], use_scaling=args['scaling'], scale_centre=args['scale_top'], exhaustive_search=args['exhaustive_search'], use_hsv=args['use_rgb'], strip_luminance=args['hsv_strip_luminance'], plot_results=args['plot_results'])

    if args['plot_results']:
        plt.show()

if __name__ == '__main__':
    main()
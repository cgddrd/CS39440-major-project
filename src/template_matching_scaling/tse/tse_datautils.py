from __future__ import division

import numpy as np

__author__ = 'connorgoddard'


class TSEDataUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def get_smallest_key_dict(dict):
        return min(dict, key=dict.get)

    @staticmethod
    def get_largest_key_dict(dict):
        return max(dict, key=dict.get)

    @staticmethod
    def get_smallest_key_value_dict(dict):
        smallest_dict_key = min(dict, key=dict.get)
        return dict[smallest_dict_key]

    @staticmethod
    def string_2d_list_to_int_2d_list(string_list):
        return map(TSEDataUtils.convert_list_to_int, string_list)

    @staticmethod
    def convert_list_to_int(list_to_convert):
        return map(int, list_to_convert)

    @staticmethod
    def calc_centered_moving_average(values, window):

        # Using 'same' mode for the 'numpy.convolve' function will centre the convolution window at each point of the array overlap.
        # WARNING: Boundary effects will be observed at the ends as a result of the arrays not fully overlapping.
        return TSEDataUtils.calc_moving_average(values, window, 'same')


    @staticmethod
    def calc_moving_average(values, window, mode='valid'):

        weights = np.repeat(1.0, window)/window

        # Including 'valid' MODE will REQUIRE there to be enough data points before beginning convolution.
        # 'valid' mode only convolve where points overlap exactly. This is equivalent to a Simple Moving Average.
        # See: http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
        return np.convolve(values, weights, mode)

    @staticmethod
    def convert_array_to_numpy_array(original_array):
        return np.array(original_array)

    @staticmethod
    def calc_cartesian_product(arrays):

        la = len(arrays)

        arr = np.empty([len(a) for a in arrays] + [la])

        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a

        return arr.reshape(-1, la)

    @staticmethod
    def calc_1d_array_average(data):
        # In single precision, mean can be inaccurate. Computing the mean in float64 is more accurate.
        # (http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean)
        return np.mean(data, dtype=np.float64)

    @staticmethod
    def numpy_array_indices_subset(data, indices_list):

        # Ensure we are dealing with a numpy array before operating.
        data = TSEDataUtils.convert_array_to_numpy_array(data)

        return data[indices_list]

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/11686764
    def filter_outliers_mean_stdev(data, stdev_factor=2):

        # Ensure we are dealing with a numpy array before operating.
        data = TSEDataUtils.convert_array_to_numpy_array(data)

        return data[abs(data - np.mean(data)) < stdev_factor * np.std(data)]

    @staticmethod
    # Modified from original source: http://stackoverflow.com/q/11686720
    def filter_outliers_mean_stdev_alternative(data, stdev_factor=2):

        # Ensure we are dealing with a numpy array before operating.
        data = TSEDataUtils.convert_array_to_numpy_array(data)

        u = np.mean(data)
        s = np.std(data)
        filtered = [e for e in data if (u - stdev_factor * s < e < u + stdev_factor * s)]
        return filtered

    @staticmethod
    # This is a convenience function for 'TSEUtils.filter_outliers_ab_dist_median_indices' to return the actual data.
    def filter_outliers_ab_dist_median(data, ab_dist_median_factor=2.):

        # Ensure we are dealing with a numpy array before operating.
        data = TSEDataUtils.convert_array_to_numpy_array(data)

        return data[TSEDataUtils.filter_outliers_ab_dist_median_indices(data, ab_dist_median_factor)]

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/16562028
    def filter_outliers_ab_dist_median_indices(data, ab_dist_median_factor=2.):

        # Ensure we are dealing with a numpy array before operating.
        data = TSEDataUtils.convert_array_to_numpy_array(data)

        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.

        # 'np.where' returns the indices of the elements that match the mask. See: http://stackoverflow.com/a/9891802
        indices = np.where(s < ab_dist_median_factor)[0]

        return indices

    @staticmethod
    def extract_tuple_elements_list(data, tuple_index):

        result = []

        for val in data:
            result.append(val[tuple_index])

        return result

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/16158798
    def calc_element_wise_average(data):

        # 'data' expected format e.g.: '[[1, 2, 3], [1, 3, 4], [2, 4, 5]]'
        return [sum(e)/len(e) for e in zip(*data)]






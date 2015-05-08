"""

Module Name: TSEDataUtils

Description: Utility module providing common data handling and statistical processing functionality
for use with single and multi-dimensional data structures.

"""

from __future__ import division

import numpy as np

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


class TSEDataUtils(object):

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def get_smallest_key_dict(dict):
        """
        Returns the smallest key from a dictionary containing integer values.

        :param dict: Dictionary of integer values.
        :return: The smallest KEY from the dictionary, not the VALUE of the smallest key.
        """
        return min(dict, key=dict.get)

    @staticmethod
    def get_largest_key_dict(dict):
        """
        Returns the largest key from a dictionary containing integer values.

        :param dict: Dictionary of integer values.
        :return: The largest KEY from the dictionary, not the VALUE of the smallest key.
        """
        return max(dict, key=dict.get)

    @staticmethod
    def get_smallest_key_value_dict(dict):
        """
        Returns the value of the smallest key from a dictionary containing integer values.

        :param dict: Dictionary of integer values.
        :return: The value of the largest key from the dictionary.
        """
        smallest_dict_key = min(dict, key=dict.get)
        return dict[smallest_dict_key]

    @staticmethod
    def string_2d_list_to_int_2d_list(original_list):
        """
        Converts a 2D list containing String values to a 2D list of integers.

        :param original_list: The 2-dimensional list of String values.
        :return: A 2-dimensional list of Int values.
        """

        # We use 'map' to perform the conversion over all elements in the extracted 1D array.
        return map(TSEDataUtils.convert_list_to_int, original_list)

    @staticmethod
    def convert_list_to_int(original_list):
        """
        Converts a 1D list containing String values to a 1D list of integers.

        :param original_list: The list of String values.
        :return: A list of Int values.
        """
        return map(int, original_list)

    @staticmethod
    def calc_centered_moving_average(values, window):
        """
        Calculates the centered moving average from a collection of Float or Integer values.

        :param values: The original list of Float or Integer values to be averaged.
        :param window: The size of the centered convolution window (i.e. the centered range of indexs either side of the curremt index to average)
        :return: A list of Float or Integer values representing the averages of the original list.
        """

        # Using 'same' mode for the 'numpy.convolve' function will centre the convolution window at each point of the array overlap.

        # WARNING: Boundary effects will be observed at the ends as a result of the arrays not fully overlapping.
        return TSEDataUtils.calc_moving_average(values, window, 'same')


    @staticmethod
    def calc_moving_average(values, window, mode='valid'):
        """
        Calculates the moving average from a collection of Float or Integer values.

        :param values: The original list of Float or Integer values to be averaged.
        :param window: The size of the centered convolution window (i.e. the centered range of indexs either side of the curremt index to average)
        :param mode: The type of convolution mode for the 'numpy.convolve' function. (Default: 'valid')
        :return: A list of Float or Integer values representing the averages of the original list.
        """

        # Create the `window' from which to select the elements to average.
        weights = np.repeat(1.0, window)/window

        # Including 'valid' MODE will REQUIRE there to be enough data points before beginning convolution.
        # 'valid' mode only convolve where points overlap exactly. This is equivalent to a Simple Moving Average.
        # See: http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
        return np.convolve(values, weights, mode)

    @staticmethod
    def convert_array_to_numpy_array(original_array):
        """
        Converts a Python 1D list to a 'Numpy' 1D array using the 'np.array' convenience function.

        :param original_array: The standard 'Python' list to be converted.
        :return: An equivalent 'Numpy' 1D array.
        """
        return np.array(original_array)

    @staticmethod
    # This code has been modified from an original source: http://stackoverflow.com/a/11146645/4768230
    def calc_cartesian_product(arrays):
        """
        Calculates the cartesian product of elements between a collection of 1D arrays.

        :param arrays: The collection of 1D arrays of which each element will be combined with each other.
        :return: The combined results of the cartesian product between all input elements.
        """

        la = len(arrays)

        arr = np.empty([len(a) for a in arrays] + [la])

        # Perform cross combination across each array in turn.
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a

        return arr.reshape(-1, la)

    @staticmethod
    def calc_1d_array_average(data_collection):
        """
        Calculates the mean over a numeric collection (Int or Float) using the 'np.mean' function.

        :param data_collection: The collection of Integers or Floats to be averaged.
        :return: Float holding the  average of all the numeric input values.
        """
        # In single precision, mean can be inaccurate. Computing the mean in float64 is more accurate.
        # (http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean)
        return np.mean(data_collection, dtype=np.float64)

    @staticmethod
    def numpy_array_indices_subset(data_collection, indices_list):
        """
        Returns a subset of values from a collection based upon a list of specified indices.

        :param data_collection: The collection of values from which to extract specific values.
        :param indices_list: The collection of indices used to select the element values to extract from the data collection.
        :return: A subset of the original data collection defined by the specified indices.
        """
        # Ensure we are dealing with a numpy array before operating.
        data_collection = TSEDataUtils.convert_array_to_numpy_array(data_collection)
        return data_collection[indices_list]

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/11686764
    def filter_outliers_mean_stdev(data_collection, stdev_factor=2):
        """
        Filters outliers from a collection of Integers or Floats by removing all values that fall outside of a defined range between the mean and standard deviation of the collection.

        :param data_collection: The collection of values to be filtered.
        :param stdev_factor: The factor between the standard deviation and the mean.
        :return: The filtered results.
        """
        # Ensure we are dealing with a numpy array before operating.
        data_collection = TSEDataUtils.convert_array_to_numpy_array(data_collection)

        return data_collection[abs(data_collection - np.mean(data_collection)) < stdev_factor * np.std(data_collection)]

    @staticmethod
    # This is a convenience function for 'TSEUtils.filter_outliers_ab_dist_median_indices' to return the actual data.
    def filter_outliers_ab_dist_median(data_collection, ab_dist_median_factor=2.):
        """
        Filters outliers from a collection of numeric values by removing all values that fall outside of a defined range between the mean and calculated median of the collection.

        :param data_collection: The collection of values to be filtered.
        :param ab_dist_median_factor: The factor between the calculated median and the mean.
        :return: The filtered results.
        """
        # Ensure we are dealing with a numpy array before operating.
        data_collection = TSEDataUtils.convert_array_to_numpy_array(data_collection)

        return data_collection[TSEDataUtils.filter_outliers_ab_dist_median_indices(data_collection, ab_dist_median_factor)]

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/16562028
    def filter_outliers_ab_dist_median_indices(data, ab_dist_median_factor=2.):
        """
        Filters outliers from a collection of numeric values by removing all values that fall outside of a defined range between the mean and calculated median of the collection.
        Returns the indices of the filtered numeric values as opposed to the values themselves.

        :param data_collection: The collection of values to be filtered.
        :param ab_dist_median_factor: The factor between the calculated median and the mean.
        :return: The indices of the filtered numeric values.
        """
        # Ensure we are dealing with a numpy array before operating.
        data = TSEDataUtils.convert_array_to_numpy_array(data)

        # Calculate the median of the input dataset.
        d = np.abs(data - np.median(data))
        mdev = np.median(d)

        # Prevent calculating a factor that is negative.
        s = d/mdev if mdev else 0.

        # 'np.where' returns the indices of the elements that match the mask. See: http://stackoverflow.com/a/9891802
        indices = np.where(s < ab_dist_median_factor)[0]

        return indices

    @staticmethod
    def extract_tuple_elements_list(tuple_collection, tuple_index):
        """
        Extracts a specified element index from each tuple within a collection.

        :param tuple_collection: The collection of tuples from which values are to be extracted.
        :param tuple_index: The specified index to be extracted from each tuple.
        :return: A collection of extracted values from each tuple in the original collection.
        """
        result = []

        for val in tuple_collection:
            result.append(val[tuple_index])

        return result

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/16158798
    def calc_element_wise_average(data_collection):
        """
        Calculates the average across corresponding element within a collection of numeric arrays.

        :param data_collection: A collection of numeric arrays.
        :return: A single array containing the element-wise averages calculated.
        """

        # 'data' expected format e.g.: '[[1, 2, 3], [1, 3, 4], [2, 4, 5]]'
        return [sum(e)/len(e) for e in zip(*data_collection)]






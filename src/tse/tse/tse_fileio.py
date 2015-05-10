"""

Module Name: TSEFileIO

Description: Provides functions relating to data file input and output (excluding images)

"""

import os
import errno

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


class TSEFileIO:

    def __init__(self, filepath="./"):
        self._file_path = filepath

    def write_tuple_list_to_file(self, filename, data_collection, data_prefix=None, append=True):
        """
        Writes the values of a collection of tuples to a file.

        :param filename: The name of the file to be written.
        :param data_collection: The collection of tuples to be written to file.
        :param data_prefix: Optional data file to be written on the first line of the export file.
        :param append: Specifies whether the exisiting file should be overwritten, or appended to.
        """

        # Check that the save file path exists, and if not, create the appropriate parent folders.
        self.check_directory(self._file_path)

        file_io_op = 'a' if append else 'w'

        new_file = open(self._file_path + filename, file_io_op)

        # If prefix has been specified, write this on the first line in the file.
        if data_prefix is not None:
            new_file.write(data_prefix + "\n")

        for data in data_collection:
            new_file.write("{0},{1}\n".format(data[0], data[1]))

        new_file.close()

    @staticmethod
    def read_file(file_path, split_delimiter=None, start_position=0):
        """
        Loads in a collection of tuple data from a configuration file.

        :param file_path: The path of the file to be read.
        :param split_delimiter: Optional delimiter for separating multiple results per line in the file.
        :param start_position: Line number to start reading from within the file.
        :return 1D collection of values loaded in from the file.
        """

        try:

            file_results = []

            with open(file_path, 'r') as open_file:

                for i, line in enumerate(open_file):

                    if i >= start_position:

                        # If a split delimter has been set, split and add all of the multiple values to the collection.
                        if split_delimiter is not None:

                            file_results.append(line.rstrip().split(split_delimiter))

                        else:

                            file_results.append(line.rstrip())

        except IOError as e:

            if e.errno == errno.ENOENT:
                raise IOError("File not found: {0}".format(file_path))

            print "Error: Unable to read data - {0}".format(file_path)

        return file_results

    @staticmethod
    # Modified from original source: http://stackoverflow.com/a/5032238
    def check_directory(directory_path):
        """
        Determines if the current directory exists (multi-platform compatible)

        :param directory_path: The path of the directory.
        """

        try:
            os.makedirs(directory_path)
        except OSError:
            if not os.path.isdir(directory_path):
                raise
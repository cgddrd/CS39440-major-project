import os

__author__ = 'connorgoddard'


class TSEFileIO:

    def __init__(self, filepath="./"):
        self._file_path = filepath

    def write_file(self, filename, data_prefix, data_collection, append):
        self.check_directory(self._file_path)

        file_io_op = 'a' if append else 'w'

        new_file = open(self._file_path + filename, file_io_op)

        new_file.write(data_prefix + "\n")

        for data in data_collection:
            new_file.write("{0},{1}\n".format(data[0], data[1]))

        new_file.close()

    @staticmethod
    def read_file(file_path, split_delimiter=None, start_position=0):

        try:

            file_results = []

            with open(file_path, 'r') as open_file:

                for i, line in enumerate(open_file):

                    if i >= start_position:

                        if split_delimiter is not None:

                            file_results.append(line.rstrip().split(split_delimiter))

                        else:

                            file_results.append(line.rstrip())

        except IOError:
            print "Error: Unable to open file or read data({0})".format(file_path)

        return file_results

    @staticmethod
    # Modified from: http://stackoverflow.com/a/5032238
    def check_directory(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError:
            if not os.path.isdir(directory_path):
                raise








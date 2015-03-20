__author__ = 'connorgoddard'
import os

class FileIO:
    def __init__(self, filepath = "./"):
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
    def check_directory(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError:
            if not os.path.isdir(directory_path):
                raise








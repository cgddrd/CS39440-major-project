__author__ = 'connorgoddard'


class FileIO:
    def __init__(self, filepath = "./"):
        self._filepath = filepath

    def write_file(self, filename, data_prefix, data_collection):
        new_file = open(self._filepath + filename, 'w')

        new_file.write(data_prefix + "\n")

        for point in data_collection:
            new_file.write("{0},{1}\n".format(point[0], point[1]))

        new_file.close()






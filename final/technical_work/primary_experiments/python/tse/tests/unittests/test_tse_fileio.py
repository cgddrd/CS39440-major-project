from unittest import TestCase
from nose.tools import *
from tse.tse_fileio import TSEFileIO

import os
import shutil

__author__ = 'connorgoddard';


class TestTSEFileIO(TestCase):

    def setUp(self):
        self.test_fileio = TSEFileIO('./data/')

    @classmethod
    def tearDownClass(cls):

        # Remove entire directory and all files.
        shutil.rmtree("./data/")

    def test_constructor(self):
        assert_equals(self.test_fileio._file_path, "./data/")

    def test_write_file(self):

        self.test_fileio.write_tuple_list_to_file("test_data.txt", [(1, 2), (3, 4), (5, 6)], "test data")

        assert_true(os.path.isfile("./data/test_data.txt"))

    def test_read_file_prefix(self):

        self.test_fileio.write_tuple_list_to_file("test_data_read_prefix.txt", [(1, 2), (3, 4), (5, 6)], "test data")

        expected_result = [['1', '2'], ['3', '4'], ['5', '6']]

        assert_equal(self.test_fileio.read_file("./data/test_data_read_prefix.txt", ",", 1), expected_result)

    def test_read_file_no_prefix(self):

        self.test_fileio.write_tuple_list_to_file("test_data_read_no_prefix.txt", [(1, 2), (3, 4), (5, 6)])

        expected_result = [['1', '2'], ['3', '4'], ['5', '6']]

        assert_equal(self.test_fileio.read_file("./data/test_data_read_no_prefix.txt", ",", 0), expected_result)

    def test_read_file_no_delimiter(self):

        self.test_fileio.check_directory("./data/")

        f = open("./data/test_file.txt", 'w')
        f.write("1\n2\n3\n")
        f.close()

        expected_result = ['1', '2', '3']

        assert_equal(self.test_fileio.read_file("./data/test_file.txt", None, 0), expected_result)

    def test_read_file_invalid_file(self):

        with assert_raises(IOError) as e:
            self.test_fileio.read_file("./data/invalid_file.txt", None, 0)

        assert_equal(e.exception.message, 'File not found: ./data/invalid_file.txt')


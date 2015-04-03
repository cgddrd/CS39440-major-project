import unittest
from nose.tools import *
from tse.tse_result import TSEResult

__author__ = 'connorgoddard'


class TestTSEResult(unittest.TestCase):
    def setUp(self):
        self.test_result = TSEResult(1, 100)

    def test_constructor(self):
        assert_equal(self.test_result.row, 1, "Row should equal 1")
        assert_equal(self.test_result.displacement, 100, "Displacement should equal 100")

    def test_set_row(self):
        assert_equal(self.test_result.row, 1, "Row should equal 1")
        self.test_result.row = 2
        assert_equal(self.test_result.row, 2, "Row should equal 2")

    def test_set_displacement(self):
        assert_equal(self.test_result.displacement, 100, "Row should equal 1")
        self.test_result.displacement = 200
        assert_equal(self.test_result.displacement, 200, "Row should equal 2")

    def test_get_tuple(self):
        assert_equal(self.test_result.to_tuple(), (1, 100), "Result value should equal (100, 200)")

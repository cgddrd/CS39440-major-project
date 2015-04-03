import unittest
from nose.tools import *
from tse.tse_point import TSEPoint

__author__ = 'connorgoddard'


class TestTSEPoint(unittest.TestCase):
    def setUp(self):
        self.test_point = TSEPoint(1, 2)

    def test_constructor(self):
        assert_equal(self.test_point.x, 1, "X should equal 1")
        assert_equal(self.test_point.x, 1, "X should equal 1")

    def test_set_x(self):
        assert_equal(self.test_point.x, 1, "X should equal 1")
        self.test_point.x = 20
        assert_equal(self.test_point.x, 20, "X should now equal 20")

    def test_set_y(self):
        assert_equal(self.test_point.y, 2, "Y should equal 2")
        self.test_point.y = 21
        assert_equal(self.test_point.y, 21, "Y should now equal 21")

    def test_get_x(self):
        assert_equal(self.test_point.x, 1, "X should equal 1")

    def test_get_y(self):
        assert_equal(self.test_point.x, 1, "X should equal 1")

    def test_get_tuple(self):
        assert_equal(self.test_point.to_tuple(), (1, 2), "Point value should equal (1, 2)")

import unittest
from nose.tools import assert_equal
from point import Point

__author__ = 'connorgoddard'


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.point = Point(1, 2)

    def test_constructor(self):
        assert_equal(self.point.x, 1, "X should equal 1")
        assert_equal(self.point.x, 1, "X should equal 1")

    def test_set_x(self):
        assert_equal(self.point.x, 1, "X should equal 1")
        self.point.x = 20
        assert_equal(self.point.x, 20, "X should now equal 20")

    def test_set_y(self):
        assert_equal(self.point.y, 2, "Y should equal 2")
        self.point.y = 21
        assert_equal(self.point.y, 21, "Y should now equal 21")

    def test_get_x(self):
        assert_equal(self.point.x, 1, "X should equal 1")

    def test_get_y(self):
        assert_equal(self.point.x, 1, "X should equal 1")

    def test_get_value(self):
        assert_equal(self.point.get_value(), (1, 2), "Point value should equal (1, 2)")

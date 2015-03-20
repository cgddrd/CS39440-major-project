from unittest import TestCase

__author__ = 'connorgoddard'

from point import Point


class TestPoint(TestCase):
    def setUp(self):
        self.point = Point(1, 2)

    def test_constructor(self):
        self.assertEqual(self.point.x, 1, "X should equal 1")
        self.assertEqual(self.point.x, 1, "X should equal 1")

    def test_set_x(self):
        self.assertEqual(self.point.x, 1, "X should equal 1")
        self.point.x = 20
        self.assertEqual(self.point.x, 20, "X should now equal 20")

    def test_set_y(self):
        self.assertEqual(self.point.y, 2, "Y should equal 2")
        self.point.y = 21
        self.assertEqual(self.point.y, 21, "Y should now equal 21")

    def test_get_x(self):
        self.assertEqual(self.point.x, 1, "X should equal 1")

    def test_get_y(self):
        self.assertEqual(self.point.x, 1, "X should equal 1")

    def test_get_value(self):
        self.assertEqual(self.point.get_value(), (1, 2), "Point value should equal (1, 2)")
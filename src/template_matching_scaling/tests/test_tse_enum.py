import unittest
import nose.tools as nt
from tse.tse_enum import enum

__author__ = 'connorgoddard'


class TestTSEEnum(unittest.TestCase):
    def setUp(self):
        self.test_enum = enum('VALUE_1', 'VALUE_2', 'VALUE_3')

    def test_constructor(self):
        nt.assert_equal(self.test_enum.VALUE_1, 0, "First enum value should equal 0")
        nt.assert_equal(self.test_enum.VALUE_2, 1, "First enum value should equal 1")
        nt.assert_equal(self.test_enum.VALUE_3, 2, "First enum value should equal 2")

    def test_not_null(self):
        nt.assert_true(self.test_enum is not None)
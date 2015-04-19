import unittest
from nose.tools import *
from tse.tse_matchtype import TSEMatchType
from tse.tse_matchmethod import tse_match_methods

__author__ = 'connorgoddard'


class TestTSEMatchType(unittest.TestCase):
    def setUp(self):
        self.test_matchtype = TSEMatchType("TestMatchType", tse_match_methods.DISTANCE, None, "r")

    def test_constructor(self):
        assert_equal(self.test_matchtype.format_string, "r", "Format string should equal 'r'")
        assert_equal(self.test_matchtype.match_name, "TestMatchType", "Match name should equal 'TestMatchType'")
        assert_equal(self.test_matchtype.match_id, None, "Match ID should equal None")
        assert_equal(self.test_matchtype.match_type, tse_match_methods.DISTANCE, "Match type should equal 'tse_match_methods.DISTANCE'")
        assert_equal(self.test_matchtype.reverse_score, False, "Reverse score (default parameter) should equal False")

    def test_constructor_default_parameter(self):

        assert_equal(self.test_matchtype.reverse_score, False, "Reverse score (default parameter) should equal False")

        # Create a new MatchType and set the default parameter (reverse_score) to True.
        self.test_matchtype2 = TSEMatchType("TestMatchType2", tse_match_methods.DISTANCE_ED, None, "b", True)
        assert_equal(self.test_matchtype2.reverse_score, True, "Reverse score should equal True")

    def test_set_match_name(self):
        assert_equal(self.test_matchtype.match_name, "TestMatchType", "Match name string should equal 'TestMatchType'")
        self.test_matchtype.match_name = "TestMatchType2"
        assert_equal(self.test_matchtype.match_name, "TestMatchType2", "Match name string should equal 'TestMatchType2'")

    def test_set_format_string(self):
        assert_equal(self.test_matchtype.format_string, "r", "Format string should equal 'r'")
        self.test_matchtype.format_string = "g"
        assert_equal(self.test_matchtype.format_string, "g", "Format string should equal 'g'")

    def test_set_match_type(self):
        assert_equal(self.test_matchtype.match_type, tse_match_methods.DISTANCE, "Match type should equal 'tse_match_methods.DISTANCE'")
        self.test_matchtype.match_type = tse_match_methods.DISTANCE_ED
        assert_equal(self.test_matchtype.match_type, tse_match_methods.DISTANCE_ED, "Match type should equal 'tse_match_methods.DISTANCE_ED'")

    def test_set_reverse_score(self):
        assert_equal(self.test_matchtype.reverse_score, False, "Reverse score should equal 'False'")
        self.test_matchtype.reverse_score = True
        assert_equal(self.test_matchtype.reverse_score, True, "Reverse score should equal 'True'")

    @raises(AttributeError)
    def test_set__invalid_match_type(self):
        self.test_matchtype.match_type = tse_match_methods.BLAH
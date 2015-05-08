"""

Module Name: TSEMatchType

Description: Provides a class representation for a single appearance-based similarity measure,
in order to store specific properties regarding each metric.

"""

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


class TSEMatchType:
    def __init__(self, match_name, match_type, match_id, format_string, reverse_score=False):
        """
        :param match_name: String representation of metric name.
        :param match_type: The 'TSEMatchMethod' enum representing the metric category.
        :param match_id: The OpenCV-specific enum representation for the similarity metric.
        :param format_string: The 'Matplotlib' configuration string used to define the format/colour of the plot line for this particular similarity measure.
        :param reverse_score: Specifies whether or not for this particular similarity a metric, a LOWER score indicates a BETTER match.
        """

        self._match_name = match_name
        self._match_type = match_type
        self._match_id = match_id
        self._format_string = format_string
        self._reverse_score = reverse_score

    @property
    def match_name(self):
        return self._match_name

    @property
    def match_type(self):
        return self._match_type

    @property
    def match_id(self):
        return self._match_id

    @property
    def format_string(self):
        return self._format_string

    @property
    def reverse_score(self):
        return self._reverse_score



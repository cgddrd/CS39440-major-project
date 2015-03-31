__author__ = 'connorgoddard'

class TSEMatchType:
    def __init__(self, match_name, match_type, match_id, format_string, reverse_score=False):
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



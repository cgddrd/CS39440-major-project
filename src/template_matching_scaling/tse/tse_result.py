__author__ = 'connorgoddard'


class TSEResult:
    def __init__(self, row, displacement, match_scores):
        self._row = row
        self._displacement = displacement
        self._match_scores = match_scores

    # Python 'toString' method for class.
    def __str__(self):
        return str(self.to_tuple())

    # Required to return a string representation while in a list. See: http://stackoverflow.com/a/727779
    def __repr__(self):
        return self.__str__()

    @property
    def row(self):
        """I'm the 'row' property."""
        return self._row

    @property
    def displacement(self):
        """I'm the 'displacement' property."""
        return self._displacement

    @property
    def match_scores(self):
        """
        'match_scores' property.
        :return:
        """
        return self._match_scores

    def to_tuple(self):
        return self.row, self.displacement, self.match_scores
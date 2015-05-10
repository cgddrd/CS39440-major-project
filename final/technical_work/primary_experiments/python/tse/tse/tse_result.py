"""

Module Name: TSEResult

Description: Provides a class representation for a single 'result' recorded within an experiment for the investigation.

"""

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


class TSEResult:
    def __init__(self, row, displacement, match_scores):
        """
        :param row: The image row upon which this result is recorded.
        :param displacement: The level of vertical displacement measured.
        :param match_scores: A collection of all the match scores calculated as part of the search.
        """

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
        return self._row

    @property
    def displacement(self):
        return self._displacement

    @property
    def match_scores(self):
        return self._match_scores

    def to_tuple(self):

        # Notice here that a tuple does not have to limited to two values only.
        return self.row, self.displacement, self.match_scores
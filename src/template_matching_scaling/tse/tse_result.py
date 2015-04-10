__author__ = 'connorgoddard'


class TSEResult:
    def __init__(self, row, displacement):
        self._row = row
        self._displacement = displacement

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

    def to_tuple(self):
        return self.row, self.displacement
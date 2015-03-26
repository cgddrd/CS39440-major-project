__author__ = 'connorgoddard'


class TSEResult:
    def __init__(self, row, displacement):
        self._row = row
        self._displacement = displacement

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
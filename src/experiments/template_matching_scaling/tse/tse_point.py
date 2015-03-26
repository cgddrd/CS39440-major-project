__author__ = 'connorgoddard'


class TSEPoint:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    # e.g. print point.x
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @property
    # e.g. print point.y
    def y(self):
        """I'm the 'y' property."""
        return self._y

    def to_tuple(self):
        return self.x, self.y
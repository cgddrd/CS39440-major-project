__author__ = 'connorgoddard'


class Point:
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

    @x.setter
    # e.g. point.x = <value>
    def x(self, value):
        self._x = value

    @y.setter
    # e.g. point.y = <value>
    def y(self, value):
        self._y = value

    def get_value(self):
        return (self.x, self.y)
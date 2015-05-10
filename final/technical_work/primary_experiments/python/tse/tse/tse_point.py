"""

Module Name: TSEPoint

Description: Provides a class representation for a single 2D-coordinate point within an image used when performing
geometric transformations of pixel coordinates.

"""

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'


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

    # Return a tuple representation of the coordinate.
    def to_tuple(self):
        return self.x, self.y

    # Python 'toString' method for class.
    def __str__(self):
        return str(self.to_tuple())

    # Required to return a string representation while in a list. See: http://stackoverflow.com/a/727779
    def __repr__(self):
        return self.__str__()
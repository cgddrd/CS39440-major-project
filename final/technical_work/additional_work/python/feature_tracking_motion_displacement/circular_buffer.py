"""
Note: This code is heavily based upon the C# implementation developed by Dr Rainer Hessmer (http://www.hessmer.org/blog/2010/08/17/monocular-visual-odometry/comment-page-1/)

"""

__author__ = 'connorgoddard'


class CircularBuffer(object):

    _next_index = 0
    _current_index = 0
    _full_count = 0

    def __init__(self, size):

        self._max_size = size
        self._history = [None] * self._max_size

    def add(self, new_value):

        if self._full_count < self._max_size:
            self._full_count += 1

        self._history[self._next_index] = new_value

        self._current_index = self._next_index

        # We use the modulo operator to "wrap around" the circular buffer.
        self._next_index = (self._next_index + 1) % self._max_size

    def get_max_size(self):
        return self._max_size

    def get_next_index(self):
        return self._next_index

    def get_full_count(self):
        return self._full_count

    def is_full(self):
        return self._full_count == self._max_size

    def get_value(self, index):

        # print "Max Size: {0}, Next Index: {1}, Full Count: {2}".format(self._max_size, self._next_index, self._full_count)

        internal_index = (self._max_size + self._next_index - self._full_count + index) % self._max_size

        return self._history[internal_index]

    def get_value_current(self, index):

        return self._history[self._current_index + index]

__author__ = 'connorgoddard'


class CircularBuffer(object):

    _next_index = 0
    _full_count = 0

    def __init__(self, size):

        self._max_size = size
        self._history = [self._max_size]

    def add(self, new_value):

        if self._full_count < self._max_size:

            self._full_count += 1

        self._history[self._next_index] = new_value

        # We use the modulo operator to "wrap around" the circular buffer.
        self._next_index = (self._next_index + 1) % self._max_size

    def get_max_size(self):
        return self._max_size

    def get_full_count(self):
        return self._full_count

    def is_full(self):
        return self._full_count == self._max_size

    def get_history_value(self, index):

        internal_index = (self._max_size + self._next_index - self._full_count + index) % self._max_size

        return self._history[internal_index]
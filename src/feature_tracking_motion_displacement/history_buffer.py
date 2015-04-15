__author__ = 'connorgoddard'

from circular_buffer import CircularBuffer

class HistoryBuffer(CircularBuffer):

    def __init__(self, history_threshold):
        CircularBuffer.__init__(self, history_threshold)

    def get_value(self, index):

        if (index > 0) or (index <= (self._max_size * -1)):

            raise IndexError("Index should be -{0} < index <= 0".format(self._max_size))

        return super(HistoryBuffer, self).get_value(self._full_count - 1 + index)





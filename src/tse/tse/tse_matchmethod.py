"""

Module Name: TSEMatchMethod

Description: Enum type used to represent the three possible matching metric categories available:
distance-based (normal), distance-based (Euclidean) and histogram-based

The two types of distance-based category are as a result of the way OpenCV handles calculation of Euclidean Distance
in a separate manner to other distance-based similarity measures.

"""

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'

from tse.tse_enum import enum

tse_match_methods = enum(
    'DISTANCE',
    'DISTANCE_ED',
    'HIST'
)

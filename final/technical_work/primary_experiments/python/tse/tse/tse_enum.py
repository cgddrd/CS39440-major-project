"""

Module Name: TSEEnum

Description: Represents the definition of an enum type for use in Python which does not come as standard in Python 2.7.

"""

__author__ = 'Connor Luke Goddard (clg11@aber.ac.uk)'

# Definition of Enum type courtesy: http://stackoverflow.com/a/1695250
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
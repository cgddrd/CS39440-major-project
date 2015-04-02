# *Estimation of Terrain Shape Using a Monocular Vision-based System* (CS39440 Major Project) [![Build Status](https://travis-ci.org/cgddrd/CS39440-major-project.svg)](https://travis-ci.org/cgddrd/CS39440-major-project) [![Stories in Ready](https://badge.waffle.io/cgddrd/cs39440-major-project.png?label=ready&title=Ready)](https://waffle.io/cgddrd/cs39440-major-project) [![Coverage Status](https://coveralls.io/repos/cgddrd/CS39440-major-project/badge.svg?branch=develop)](https://coveralls.io/r/cgddrd/CS39440-major-project?branch=develop)

## Project Metrics

### Average Throughput 
[![Throughput Graph](https://graphs.waffle.io/cgddrd/cs39440-major-project/throughput.svg)](https://waffle.io/cgddrd/cs39440-major-project/metrics)

## Background

Accurate navigation allows for robot requiring autonomous motion capabilities to make safe, yet objective
decisions on how best to traverse from a starting location to a target location over typically uneven
and/or poorly modelled terrain.

The identification and subsequent avoidance of obstacles is naturally a crucial ability for an autonomous
mobile robot to possess in order to help to maximise its own chances of survival.

Combining recent advances in camera technology with appropriate computer vision algorithms and
technique, the proposed project aims to design and implement a **vision-based software application capable
of estimating the general “shape” of the terrain currently in front of a moving robot as it follows
a route through its environment**. 

Through this system, it should be possible to identify the presence of
both positive, and negative obstacles (e.g. rocks and pits respectively), providing a reasonable indication
of their general size and location.

In addition, it is predicted that such a system will also be able to provide an estimation of the speed
and change in rotation/orientation of the robot as it traverses along a path. These will be calculated as
by-products of the terrain inference mechanism, and could form part of a larger visual odometry system.

While the primary aims of the proposed project are research-focussed, the ultimate goal of the project
will be to implement the system onto a working mobile robot, such as the ‘IDRIS’ all-terrain wheeled
robot currently in use by the Aberystwyth University Computer Science department.

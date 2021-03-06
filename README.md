# *Estimation of Terrain Shape Using a Monocular Vision-based System* (CS39440 Major Project) 
[![Build Status](https://travis-ci.org/cgddrd/CS39440-major-project.svg)](https://travis-ci.org/cgddrd/CS39440-major-project) [![Stories in Ready](https://badge.waffle.io/cgddrd/cs39440-major-project.png?label=ready&title=Ready)](https://waffle.io/cgddrd/cs39440-major-project) [![Coverage Status](https://coveralls.io/repos/cgddrd/CS39440-major-project/badge.svg?branch=develop)](https://coveralls.io/r/cgddrd/CS39440-major-project?branch=develop) [![Code Climate](https://codeclimate.com/github/cgddrd/CS39440-major-project/badges/gpa.svg)](https://codeclimate.com/github/cgddrd/CS39440-major-project) 

## Background

Accurate navigation allows for robot requiring autonomous motion capabilities to make safe, yet objective
decisions on how best to traverse from a starting location to a target location over typically uneven
and/or poorly modelled terrain.

The identification and subsequent avoidance of obstacles is naturally a crucial ability for an autonomous
mobile robot to possess in order to help to maximise its own chances of survival.

Combining recent advances in camera technology with appropriate computer vision algorithms and
technique, the proposed project aims to design and implement a **vision-based software application capable
of estimating the gradient conditions of the terrain currently in front of a moving robot as it follows
a route through its environment**.

Through this system, it should be possible to identify the presence of
both positive, and negative obstacles (e.g. rocks and pits respectively), providing a reasonable indication
of their general size and location.

In addition, it is predicted that such a system will also be able to provide an estimation of the speed
and change in rotation/orientation of the robot as it traverses along a path. These will be calculated as
by-products of the terrain inference mechanism, and could form part of a larger visual odometry system.

## Installation

**Please note:** Instructions should be followed at a **project-level** (i.e. not at repo root). Please ```cd``` into the appropiate project folder within the `src` folder.

1. Install dependencies using **pip**.

    **Please note:** Due to Cython requiring installation **prior to running** `setup.py`, this step must be followed (i.e. do not simply try to install dependencies as part of the `setup.py` installation).

        pip install -r requirements.txt

2. Compile 'C' extension modules (`.pyx`) using [Cython](https://github.com/cython/cython).

        python setup.py build_ext --inplace

3. Install project module.

        python setup.py install

from __future__ import division

__author__ = 'connorgoddard'

import numpy as np
import cv2

width = 100
height = 50

width2 = 150
height2 = 75

windowScale = 5.0

scalePercent = 1.1

scaleUnit = (width + 1) / float(width)

# Allocating a matrix to represent our "black image". The '3' is to allocate enough room for three channels (RGB).
img = np.zeros((height2 * windowScale, width2 * windowScale, 3), np.uint8)

originX = int((width2 * (windowScale / 2)) - (width / 2))
originY = int((height2 * (windowScale / 2) - (height / 2)))

topLeft = (originX, originY)
topRight = (originX + width, originY)
bottomLeft = (originX, originY + height)
bottomRight = (originX + width, originY + height)
centreLeft = (originX, originY + (height / 2))
centreCentre = (originX + (width / 2), originY + (height / 2))
centreRight = (originX + width, originY + (height / 2))
bottomCentre = (originX + (width / 2), originY + height)
topCentre = (originX + (width / 2), originY)

oldPoints = [topLeft, topCentre, topRight, centreLeft, centreCentre, centreRight, bottomLeft, bottomCentre, bottomRight]

originX2 = int((width2 * (windowScale / 2)) - (width2 / 2))
originY2 = int((height2 * (windowScale / 2)) - (height2 / 2))

topLeft2 = (originX2, originY2)
topRight2 = (originX2 + width2, originY2)
bottomLeft2 = (originX2, originY2 + height2)
bottomRight2 = (originX2 + width2, originY2 + height2)
centreLeft2 = (originX2, originY2 + (height2 / 2))
centreCentre2 = (originX2 + (width2 / 2), originY2 + (height2 / 2))
centreRight2 = (originX2 + width2, originY2 + (height2 / 2))
bottomCentre2 = (originX2 + (width2 / 2), originY2 + height2)
topCentre2 = (originX2 + (width2 / 2), originY2)

oldPoints2 = [topLeft2, topCentre2, topRight2, centreLeft2, centreCentre2, centreRight2, bottomLeft2, bottomCentre2, bottomRight2]

newScaleFactor = float(width2 / width)

for s in oldPoints:

    cv2.circle(img, (int(s[0]), int(s[1])), 2, (255, 0, 0), 1, 8, 0)

    vec = (s[0] - centreCentre[0], s[1] - centreCentre[1])

    newVec = ((vec[0] * newScaleFactor), (vec[1] * newScaleFactor))

    cv2.circle(img, (int(newVec[0] + centreCentre[0]), int(newVec[1] + centreCentre[1])), 2, (0, 255, 0), 1, 8, 0)

for t in oldPoints2:

    cv2.circle(img, (int(t[0]), int(t[1])), 2, (255, 0, 255), 1, 8, 0)


cv2.imshow("output", img)
k = cv2.waitKey(0)

if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
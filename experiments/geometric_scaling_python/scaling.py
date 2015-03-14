import numpy as np
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('../eval_data/motion_images/TEST.JPG',0)

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

width = 100
height = 50

windowScale = 5.0

scalePercent = 1.1

scaleUnit = (width + 1) / float(width)

print scaleUnit

# Allocating a matrix to represent our "black image". The '3' is to allocate enough room for three channels (RGB).
img = np.zeros((height * windowScale, width * windowScale, 3), np.uint8)

originX = int((width * (windowScale / 2)) - (width / 2))
originY = int((height * (windowScale / 2)) - (height / 2))

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

for s in oldPoints:

    cv2.circle(img,s, 2, (0,0,255), 1, 8, 0)

    vec = np.array(s) - np.array(centreCentre)

    scalePercentResultPoint = (vec * scalePercent)
    # scaleUnitResultPoint = (vec * scaleUnit)

    cv2.circle(img,totuple(scalePercentResultPoint.astype(int) + np.array(centreCentre)), 2, (255,128,0), 1, 8, 0)

    # cv2.circle(img,totuple(np.ceil(scalePercentResultPoint).astype(int) + np.array(centreCentre)), 2, (0,255,0), -1)
    # cv2.circle(img,totuple(np.ceil(scalePercentResultPoint)  + np.array(centreCentre)), 2, (255,0,0), -1)

    # print scalePercentResultPoint
    # print "\n\n"
    # print scalePercentResultPoint.astype(int)
    # print "\n\n"
    # print np.ceil(scalePercentResultPoint)

print "\n\n"

# x = np.array(topLeft)
# y = np.array(topRight)
# z = x + y

# print topLeft
# print topRight
# print z[0]

# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# print scaleUnit

cv2.imshow("output", img)
k = cv2.waitKey(0)

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

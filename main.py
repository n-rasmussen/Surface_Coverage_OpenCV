import numpy as np
import cv2 as cv
import matplotlib as plt

# Open Image and read it into uint8 data type (BGR data) & resize image to fit on screen
img = cv.imread('pPVA4.jpg')
print(img.shape)
img = cv.resize(img, (round(.25*img.shape[1]), round(.25*img.shape[0])))
assert img is not None,  "Did not read Correctly"


# properties of image
px = img[100, 100]
print(px)
print(img.shape)
print(img.size)
print(img.dtype)


# convert image to gray scale and display it
new = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("new", new)
# cv.waitKey(0)

#using Canny edge detection.
edges = cv.Canny(new, 190, 200) # threshold for edge linking
cv.imshow("Edge detection", edges)
cv.imwrite("Edge_detection.png", edges)
cv.imshow("img", img)
cv.imwrite("img.png", img)


# dilate the edges so they are more defined.
kernel = np.ones((2, 2))
imgDil = cv.dilate(edges, kernel, iterations = 3)
imgThre = cv.erode(imgDil, kernel, iterations = 3)
# cv.imshow("Dil", imgDil)
# cv.imshow("Erod", imgThre)

# Project the contour lines from the Canny method onto the original image
contours, 	hierarchy = cv.findContours(imgThre, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0,255,0), 3)
# cv.imshow("w/ contours", img)

# make a list of the large contours so that the gel contour can be manually found.
a = []
cont = []
for con in contours:
    i = 0
    area = cv.contourArea(con)
    a.append(area)
    i += 1
    if area > 5000:
        perimeter = cv.arcLength(con, True)

        # smaller epsilon -> more vertices detected [= more precision]
        epsilon = 0.0002 * perimeter
        # check how many vertices
        approx = cv.approxPolyDP(con, epsilon, True)
        # print(len(approx))

        cont.append([len(approx), area, approx, con])

print(max(a))
print("---\nfinal number of contours: ", len(cont))


# Removing Background
# Get Dimensions
hh, ww = img.shape[:2]

# draw white contour on black background as mask
mask = np.zeros((hh,ww), dtype=np.uint8)

#contour number that represents hydrogel.
num = 3
cv.drawContours(mask,[cont[num][2]], -1, (255,255,255), cv.FILLED)

# apply mask to image
image_masked = cv.bitwise_and(img, img, mask=mask)

# convert to HSV colouring
hsv = cv.cvtColor(image_masked, cv.COLOR_BGR2HSV)
# set lower and upper color limits
lowerVal = np.array([30,100 ,50])
upperVal = np.array([100,255,200])
# Threshold the HSV image to get only red colors
mask2 = cv.inRange(hsv, lowerVal, upperVal)
# apply mask to original image
final = cv.bitwise_and(hsv, hsv, mask=mask2)

# gray final image after applying mask
gray = cv.cvtColor(final, cv.COLOR_BGR2GRAY)
algae_area = cv.countNonZero(gray)
print("algae_area")
print(algae_area)
algae_coverage = algae_area / cont[num][1] * 100
print("Aglae Coverage %")
print(algae_coverage)


dst = cv.add(image_masked, final)
cv.imshow("final", final)
cv.imwrite("final.png", final)
cv.imshow("hsv", hsv)
cv.imwrite("hsv.png", hsv)
cv.imshow("over", dst)
cv.imwrite("over.png", dst)
cv.waitKey(0)
import cv2
import numpy as np

#from PIL import Image


# *******************WEBSITE LINKS*******************

#https://towardsdatascience.com/find-where-to-park-in-real-time-using-opencv-and-tensorflow-4307a4c3da03
#https://deeplearninganalytics.org/blog/coding-a-parking-spot-detector
#https://github.com/priya-dwivedi/Deep-Learning/blob/master/parking_spots_detector/identify_parking_spots.ipynb

#*****************************************************


im = cv2.imread('greyscale.jpg')

# apply canny set upper and lower thresholds using formula:(https://en.wikipedia.org/wiki/Otsu%27s_method)
# apply automatic Canny edge detection using the computed median from that^ formula
v = np.median(im)
lower = int(max(0, (1.0 - .33) * v))
upper = int(min(255, (1.0 + .33) * v))
edged = cv2.Canny(im, lower, upper)
img = cv2.resize(edged, (960, 760))
#cv2.imshow('Edged', edged)


# Select ROI / crop selected image
r = cv2.selectROI(img)
roi = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]


# select multiple ROIs and crop the image
#r = cv2.selectROIs("Select ROIs", img, fromCenter=False)
#roi1 = img[r[0][1]:r[0][1]+r[0][3], r[0][0]:r[0][0]+r[0][2]]



# Display cropped image and save it as a new .jpg file

#cv2.imshow("Image", roi1)
cv2.imshow("Image 1", roi)
cv2.imwrite("ROI.jpg", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
mask = cv2.imread('ROI.jpg')

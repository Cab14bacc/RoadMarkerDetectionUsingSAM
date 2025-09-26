import cv2
import numpy as np


img = cv2.imread("0004_cropped.tif")

img = cv2.resize(img, (np.array(img.shape[:2]) * 0.3).astype(np.int32))
cv2.imwrite("0004_cropped.png", img)
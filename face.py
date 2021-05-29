import cv2
import numpy as np

# minRange for min skin color Rnage
# maxRange for maximum skin color Range
minRange = np.array([0,133,77],np.uint8)
maxRange = np.array([235,173,127],np.uint8)


image = cv2.imread("rp.jpg")
image = cv2.resize(image, (1000,650), interpolation=cv2.INTER_CUBIC)

# change our image bgr to ycr using cvtcolor() method 
YCRimage = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)

# apply min or max range on skin area in our image
skinArea = cv2.inRange(YCRimage,minRange,maxRange)
detectedSkin = cv2.bitwise_and(image, image, mask = skinArea)

'''
(cnts, _) = cv2.findContours(skinArea.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

counter = 0
for c in cnts:
    counter = counter+1
    area = cv2.contourArea(c)
    arclen = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * arclen, True)
    

    if(counter == 1):
        pa = area
    elif(area>pa):
        pa = area
        ap = approx

cv2.drawContours(skinArea, [ap], -1, (255, 0, 255), 4)
'''

cv2.imshow("detected_skin",detectedSkin)
k = cv2.waitKey()
if k == 27:
    # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('final.jpg',detectedSkin)
    cv2.destroyAllWindows()

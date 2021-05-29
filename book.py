import cv2
import numpy as np

img = cv2.imread('Bookshelf.jpg')
img_size = cv2.resize(img, (1000,650), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img_size,cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5), np.float32)/25
convolution = cv2.filter2D(gray, -1, kernel)

#blur=cv2.medianBlur(convolution,5)
#blur=cv2.blur(convolution,(6,6))
blur = cv2.GaussianBlur(convolution, (5, 5), 0)
cv2.imshow('smothing_blurred_gray',blur)
k=cv2.waitKey()

if k == 27:
    # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('smothing_blurred_gray.jpg',gray)
    cv2.destroyAllWindows()

edges = cv2.Canny(blur,50,150,apertureSize = 3)
cv2.imshow('edges',edges)
k=cv2.waitKey()

if k == 27:
    # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('edge.jpg',edges)
    cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Spaces_filled", closed)
k=cv2.waitKey()

if k == 27:
    # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('Spaces_filled.jpg',closed)
    cv2.destroyAllWindows()

'''minLineLength = 14
maxLineGap = 5

lines = cv2.HoughLinesP(edge ,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img_size,(x1,y1),(x2,y2),(0,255,0),2)
'''

lines = cv2.HoughLines(edges ,1,np.pi/180,130)
for x in range(0, len(lines)):
    for rho,theta in lines[x]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_size,(x1,y1),(x2,y2),(0,255,0),2)

print(len(lines))
cv2.imshow('hough',img_size)
k=cv2.waitKey()

if k == 27:
    # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('final.jpg',img_size)
    cv2.destroyAllWindows()

vertical = 0
horizontal = 0

for i in range(0,len(lines)):
    for rho,theta in lines[i]:
        #print(theta)
        if(theta>0.87):
            vertical = vertical +1
        else:
            horizontal = horizontal + 1

print("Number of books:")
print((vertical - 2 )+(horizontal - 2))  
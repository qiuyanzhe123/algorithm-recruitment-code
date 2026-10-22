import cv2
qyz=cv2.imread("test.jpg")
grey_qyz=cv2.cvtColor(qyz,cv2.COLOR_BGR2GRAY)
_,binary_qyz=cv2.threshold(grey_qyz,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
edge=cv2.Canny(grey_qyz,50,150)
contours,_=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     
result_qyz=qyz.copy()
cv2.drawContours(result_qyz,contours,-1,(0,255,0),2)
total_area=0
total_perimeter=0
for contour in contours:
        area=cv2.contourArea(contour)
        perimeter=cv2.arcLength(contour,True)
        total_area+=area
        total_perimeter+=perimeter
print("轮廓总面积：",total_area)
print("轮廓总周长：",total_perimeter)
cv2.imwrite("test2_3.jpg",result_qyz)


import cv2
import numpy as np
qyz=cv2.VideoCapture("example2.mp4")
mog2_qyz=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=16,detectShadows=False)
width_qyz=int(qyz.get(cv2.CAP_PROP_FRAME_WIDTH))
height_qyz=int(qyz.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_qyz=qyz.get(cv2.CAP_PROP_FPS)
writer_qyz=cv2.VideoWriter("test2_4.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps_qyz,(width_qyz,height_qyz),False)
kernel_qyz=np.ones((3,3),np.uint8)
while True:
    ret_qyz,frame_qyz=qyz.read()
    if not ret_qyz:
        break
    mask_qyz=mog2_qyz.apply(frame_qyz)
    mask_qyz=cv2.medianBlur(mask_qyz,5)
    mask_qyz=cv2.morphologyEx(mask_qyz,cv2.MORPH_OPEN,kernel_qyz)
    writer_qyz.write(mask_qyz)
    cv2.imshow("MOG2 Motion Detection",mask_qyz)
    if cv2.waitKey(30)==ord("q"):
        break
qyz.release()
writer_qyz.release()
cv2.destroyAllWindows()
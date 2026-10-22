import cv2
qyz=cv2.VideoCapture("example2.mp4")
ret_qyz,prev_qyz=qyz.read()
prev_gray_qyz=cv2.cvtColor(prev_qyz,cv2.COLOR_BGR2GRAY)
width_qyz=int(qyz.get(cv2.CAP_PROP_FRAME_WIDTH))
height_qyz=int(qyz.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_qyz=qyz.get(cv2.CAP_PROP_FPS)
writer_qyz=cv2.VideoWriter("test2_4.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps_qyz,(width_qyz,height_qyz),False)
while True:
    ret_qyz,frame_qyz=qyz.read()
    if not ret_qyz:
        break
    gray_qyz=cv2.cvtColor(frame_qyz,cv2.COLOR_BGR2GRAY)
    diff_qyz=cv2.absdiff(prev_gray_qyz,gray_qyz)
    _,binary_qyz=cv2.threshold(diff_qyz,30,255,cv2.THRESH_BINARY)
    writer_qyz.write(binary_qyz)
    cv2.imshow("Basic Motion Detection",binary_qyz)
    prev_gray_qyz=gray_qyz.copy()
    if cv2.waitKey(30)==ord("q"):
            break
qyz.release()
writer_qyz.release()
cv2.destroyAllWindows()
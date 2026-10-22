import cv2
qyz=cv2.VideoCapture("example4.mp4")
qyz.set(cv2.CAP_PROP_POS_MSEC,500)
ret_qyz,frame_qyz=qyz.read()
bbox_qyz=cv2.selectROI("Select Ball",frame_qyz,False)
cv2.destroyWindow("Select Ball")
tracker_qyz=cv2.TrackerCSRT_create()
tracker_qyz.init(frame_qyz,bbox_qyz)
width_qyz=int(qyz.get(cv2.CAP_PROP_FRAME_WIDTH))
height_qyz=int(qyz.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_qyz=qyz.get(cv2.CAP_PROP_FPS)
writer_qyz=cv2.VideoWriter("test2_5_2.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps_qyz,(width_qyz,height_qyz))
while True:
    ret_qyz,frame_qyz=qyz.read()
    if not ret_qyz:
        break
    success_qyz,bbox_qyz=tracker_qyz.update(frame_qyz)
    if success_qyz:
        x_qyz,y_qyz,w_qyz,h_qyz=map(int,bbox_qyz)
        cv2.rectangle(frame_qyz,(x_qyz,y_qyz),(x_qyz+w_qyz,y_qyz+h_qyz),(0,255,0),2)
    writer_qyz.write(frame_qyz)
    cv2.imshow("1",frame_qyz)
    if cv2.waitKey(30)==ord("q"):
        break
qyz.release()
writer_qyz.release()
cv2.destroyAllWindows()
import cv2

# read frames from view
videoCapture = cv2.VideoCapture()
videoCapture.open('./video/hcz_vid.mp4')

fps = videoCapture.get(cv2.CAP_PROP_FPS)
frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
#fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
print("fps=",fps,"frames=",frames)

for i in range(int(frames / 10)):
    ret,frame = videoCapture.read()
    #cv2.imwrite("./out/hcz_vid_frame_%d.jpg"%i,frame)

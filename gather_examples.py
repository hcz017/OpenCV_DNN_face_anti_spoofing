# 导入相关的包
import numpy as np
import argparse
import pyrealsense2 as rs
import cv2
import os
import time

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
	help="# of frames to skip before applying face detection")
ap.add_argument("-f", "--flag", type=str, required=True,
    help="gather True or False frames")
args = vars(ap.parse_args())
# 加载人脸检测器 
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt.txt"])
modelPath = os.path.sep.join([args["detector"],
"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# 打开Intel RealSense Camera并初始化 
print("[INFO] Initialize Intel  RealSense camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# 设定depth图像colormap
colorizer = rs.colorizer()
colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
colorizer.set_option(rs.option.color_scheme, 2) 
pipeline.start(config)
time.sleep(2.0)
# 初始化两个变量
read = 0
saved = 0
# 循环处理帧图像
while True:
# 从Camera获取图像 
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
   	    continue
    # 将图像转为array
    color_image = np.asanyarray(color_frame.get_data())
    # 将深度彩色图转为深度灰度图（8-bit）
    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # 获取帧维度并将其转换成blob
    (h, w) = color_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    # 将blob输入到网络中，获取预测结果
    net.setInput(blob)
    detections = net.forward()
    dis_list = []
    # 已读帧数加1
    read += 1
    # 是否跳过当前帧
    if read % args["skip"] != 0:
        continue
    # 获取检测到的置信度最高的对象 
    if(len(detections)>0): 
        i = np.argmax(detections[0, 0, :, 2]) 
        # 获取置信度 
        confidence = detections[0, 0, i, 2] 
        # 判断置信度是否满足要求 
        if confidence > args["confidence"]:
            # 计算人脸检测框坐标得到人脸ROI 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # 确定检测框未超出帧大小
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            # 提取人脸ROI并进行预处理
            cv2.rectangle(color_image, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
            cv2.imshow("color_image", color_image)
            face = depth_colormap[startY:endY, startX:endX]
            if args["flag"] == "real":
                # 将图像存入磁盘（real face）
                p = os.path.sep.join([args["output"],
                  "real_face_{}.jpg".format(saved)])
            else:
                # 将图像存入磁盘（fake face）
                p = os.path.sep.join([args["output"],
                  "fake_face_{}.jpg".format(saved)])

            cv2.imshow(args["flag"], face)
            cv2.waitKey(1000)
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))
# 清理内存
vs.release()
cv2.destroyAllWindows()




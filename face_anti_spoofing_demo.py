# 导入必要的库
import numpy as np
import argparse
import pickle
import cv2
import pyrealsense2 as rs
import os
import time

# 定义命令行参数列表
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-em", "--embedder", type=str, required=True,
                help="path to recognizer")
ap.add_argument("-e", "--embeddings", required=True,
                help="path to serialized db of facial embeddings")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 加载人脸检测器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt.txt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD);
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL);
# 加载人脸身份验证器
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNet(args["embedder"] + ".bin", args["embedder"] + ".xml")
embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE);
# embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD);
# embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL);
# 加载活体检测器
print("[INFO] loading liveness detector...")
model = cv2.dnn.readNet(args["model"] + ".bin", args["model"] + ".xml")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE);
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD);
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL);
# 加载人脸身份信息数据库
data = pickle.loads(open(args["embeddings"], "rb").read())
# 初始化D415并预热
print("[INFO] Initialize Intel  Realsense camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
colorizer = rs.colorizer()
colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
colorizer.set_option(rs.option.color_scheme, 2)  # white to black
profile = pipeline.start(config)
time.sleep(2.0)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)
# 循环处理视频流
while True:
    # 从Camera获取图像
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        continue

    # 将彩色RGB图像转为array
    color_image = np.asanyarray(color_frame.get_data())
    # 将对齐后的深度彩色图转为深度灰度图（8-bit）
    depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    # 获取帧维度并将其转换成blob
    (h, w) = color_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # 将blob输入到网络中，获取预测结果
    net.setInput(blob)
    detections = net.forward()
    dis_list = []  # embedding相似度变量
    display_image = color_image.copy()

    # 循环处理检测到的人脸
    for i in range(0, detections.shape[2]):
        # 获取置信度
        confidence = detections[0, 0, i, 2]
        # 滤去部分低置信度结果
        if confidence > args["confidence"]:
            # 计算人脸检测框坐标得到人脸ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 确定检测框未超出帧大小
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            # 提取人脸RGB ROI并进行预处理
            rgb_face = color_image[startY:endY, startX:endX]
            rgb_faceBlob = cv2.resize(rgb_face, (128, 128))
            rgb_faceBlob = rgb_faceBlob.transpose(2, 0, 1)
            rgb_faceBlob = np.expand_dims(rgb_faceBlob, axis=0)
            # 将blob输入到人脸身份验证器中获取embedding
            embedder.setInput(rgb_faceBlob)
            vec = embedder.forward()[0, :, 0, 0]
            vec = np.array(vec)
            # 计算embedding之间的相似度
            for item in enumerate(data["embeddings"]):
                xy = np.dot(vec, item[1])
                xx = np.dot(vec, vec)
                yy = np.dot(item[1], item[1])
                norm = np.sqrt(xx) * np.sqrt(yy)
                dis = 1.0 - xy / norm
                dis_list.append(dis)
            j = np.argmin(dis_list)
            # 获取人脸name标签并显示
            if dis_list[j] < 0.5:
                name = data["names"][j]
                cv2.putText(display_image, name, (endX - 30, endY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            dis_list.clear()

            # 提取人脸RGB ROI并进行预处理
            depth_face = depth_colormap[startY:endY, startX:endX]
            depth_face = cv2.cvtColor(depth_face, cv2.COLOR_BGR2RGB);
            depth_faceBlob = cv2.dnn.blobFromImage(cv2.resize(depth_colormap, (224, 224)), 1.0 / 255)

            # 将blob输入到活体检测器中获取检测结果
            model.setInput(depth_faceBlob)
            preds = model.forward()[0, 0:2]
            print(preds)
            k = np.argmax(preds)

            # 绘制检测结果
            if k == 1:
                cv2.putText(display_image, "True", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(display_image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
            if k == 0:
                cv2.putText(display_image, "False", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(display_image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

    # 将带标签的图像显示在窗口中
    cv2.imshow("result", display_image)

    # 按“q”退出程序
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# 清空内存
cv2.destroyAllWindows()

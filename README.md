# 目录结构
.
├── README
├── models 训练好的网络模型
│   ├── fas 活体检测器
│   ├── fd 人脸检测器
│   └── fr 人脸身份验证器
├── train  训练FeatherNet 的相关文件
│   ├── cfgs FeatherNet 配置文件目录
│   │   └── FeatherNet.yaml FeatherNet 配置文件
│   ├── checkpoints 训练模型checkpoints 目录
│   ├── models FeatherNet 网络架构文件目录
│   │   └── FeatherNet.py FeatherNet 框架文件
│   └── train_FeatherNet.py 该脚本将训练FeatherNet 分类器。我们将使用PyTorch 来搭建并训练模型。
├── datasets 自行采集的活体检测模型训练数据集目录
│   ├── fake 翻拍视频攻击人脸同选购
│   └── real 帧是人脸图像
├── peoples 用于建立人脸身份信息数据库的人物图片
├── embeddings 保存人脸身份信息数据库的embedding(嵌入)
├── env.yaml
├── gather_examples.py 通过摄像头抓取人脸ROI 获取人脸深度想，帮助我们创建一个深度学习人脸活体检测网络训练数据集
├── read_data.py 读取数据集图像用于训练
├── register.py 该脚本用于等级人脸身份信息，建立人脸身份信息库
├── demo.py 实时支付及人脸识别演示程序脚本
├── requirements.txt

# 8.3.1 数据集准备

## 1. 活体检测模型训练数据集

1. 采集真实人脸数据。同时获得RGB 和深度图像， 在RGB 图上应用人脸检测器得到人脸ROI ，使用这个ROI 应用在深度图像上得到深度人脸，保存为real_face_xx。
2. 采集伪造人脸数据。使用相同人像的视频（手持），对准camera 采集图像，得到fake_face。
```shell
python gather_examples.py -d models/fd -o datasets/real -f real
```

## 2. 人脸身份信息数据库

将待登记的人物照片提取人脸embedding 特征向量和对应人脸身份name标签并保存，用于后续人脸信息比对。
1. 将已有的以人物名字命名的人物信息图片放入peoples/文件夹中，
2. 然后通过不同的人物图像构造人脸人分信息数据库。
```shell
python register.py -i peoples/ -d models/fd -e embeddings/embeddings -m models/fr/face-reidentification-retail-0095
```
# 8.3.2 活体检测模型训练
准备file list 和labels
参考datasets/ prepare_file_and_labels_list.sh 脚本
文件 train/train_FeatherNet.py 需要移动到根目录执行
```shell
python train_FeatherNet.py --b 32 --lr 0.01 --every-decay 60 
```

```shell
python demo.py -m models/fas/feathernetB -d models/fd -em models/fr/face-reidentification-retail-0095 -e embeddings/embeddings
```
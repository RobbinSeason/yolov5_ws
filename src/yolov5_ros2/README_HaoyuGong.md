## self readme start #
## object detection part

## 0. 安装依赖

首先，确保您已经更新了系统并且安装了必要的依赖。以下是一些安装步骤，其中`$ROS_DISTRO` 是您的ROS2发行版（例如：`foxy`、`galactic`）：

```bash
sudo apt update
sudo apt install python3-pip ros-$ROS_DISTRO-vision-msgs
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple yolov5  
```
## 1. ducument configuration
 best.pt :  need to be put under the document of config

## 2. adjust the parameter and select the camera
yolov5_ros2_launch.py :
# first node 
some parameters
# others
topic about camera

## 3. colcon build and lauch

编译项目并设置环境变量

```bash
cd yolov5_ws
colcon build
source install/setup.bash

ros2 launch yolov5_ros2 yolov5_ros2_launch.py
```

## 4.result
the result will be shown by topic : /Vision_result_Team3

basic introduction about the msg:  (only list the key-info)
---
frame_id: spot/body       # name of coorderate base
class_id: ball            # class
score: 0.8171692490577698 # the confidence of the bbox
position:                 # coorderate after tf
  x: -1.1204196044181618
  y: -0.49767466398206756
  z: -0.5263708318374491


# 简介

一个机器学习小项目，基于yolo算法，输入sar图像，进行图像中飞机的目标识别。数据集采用雷达学报上公开的[SAR-AIRcraft-1.0](https://radars.ac.cn/web/data/getData?newsColumnId=f896637b-af23-4209-8bcc-9320fceaba19)，整个流程包括数据集处理、机器学习以及可视化验收。

# 项目运行说明

## 环境部署

### cuda驱动配置

首先进行驱动的安装，进入[nvidia驱动官网](https://www.nvidia.cn/drivers/lookup/)下载安装自己显卡型号对应的驱动，建议安装最新的cuda版本

### conda环境配置

首先打开清华镜像的[链接](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.10-1-Windows-x86_64.exe)下载并安装Anaconda

之后进入Anaconda终端后在Anaconda中创建环境

```
conda -n sar-yolo python=3.10
```

弹出提示输入y并回车，等待下载安装

显示**done**后，激活环境

```
conda activate sar-yolo
```

在终端中输入以下命令

```
nvidia-smi
```

查看输出中右上方的**CUDA Version**

记住这个数字，打开[此链接](https://pytorch.org/)

下拉到**Install PyTorch**板块

依次点击Stable→Windows→Conda→Python（根据你自己具体情况改）

再找到个不高于你记下的**CUDA Version**的值点击，复制**Run this Command:**里的命令

到激活的conda终端环境中运行这个命令

同样出现提示时输入y并回车

显示**done**后，在终端中进入项目文件夹，运行以下命令

```
pip install -r requirements.txt
```

至此环境部署完毕，可进行项目运行

## 项目流程

> 注意，如果只需要进行效果验收，可以跳过步骤2，并且在步骤4中遵照验收步骤进行

> 注意：数据集需要额外下载，解压出origin_data文件夹，放到到本项目app文件夹下，才可以执行后续步骤

> 注意：本项目默认情况下使用yolov5su.pt预训练模型，使用其他模型请在config.py下更改

> 注意：网络环境影响可能在训练模型的过程中遇到错误，比如缺失字体且下载失败，根据报错提示将项目中的Arial.ttf移动到指定位置即可解决，不作赘述

1. 数据处理（必须）

python运行1_data_process.py

2. 模型训练和验证（可选）

python运行2_train&val.py

3. 测试集打标签（必须）

python运行3_test_labeler.py

4. 启动web程序（必须）

仅想进行成果验收的：

无需进行操作，直接运行即可

需要进行自行训练的：

首先注释掉config.py中的

复制第2步生成的权重文件路径，一般在runs/detect/train/weights/best.pt

复制粘贴到config.py中的BEST_PT赋值中，如果只训练了一次就不需要进行更改

最后，python运行4_web_server.py，在浏览器中输入localhost:8587并访问

在输入框传入转换后的datasets文件夹中images\test下的任意图片，上传完后点击网页的识别，静等片刻即可查询结果
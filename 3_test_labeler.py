import os
import cv2
import yaml
import numpy as np
from config import BASE_DIR

print("正在为测试集打标签....")

data_yaml = os.path.join(BASE_DIR, "app", "data.yaml")
# 加载配置文件
with open(data_yaml, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 路径设置
image_folder = os.path.join(
    BASE_DIR, "app", config["test"]
)  # 测试图像文件夹
label_folder = os.path.join(
    BASE_DIR, "app","datasets", "labels", "test"
)  # 测试标签文件夹

output_folder = os.path.join(BASE_DIR, "app", "origin_test_wirth_labels")

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 类别列表
classes = config["names"]

# 定义颜色列表
colors = np.random.randint(0, 255, size=(len(classes), 3)).tolist()

# 遍历所有图像文件
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):  # 根据你的图像格式
        # 读取图像
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        # 读取相应的标签文件
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_folder, label_file)

        # 检查标签文件是否存在
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 解析标签
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.split())
                class_id = int(class_id)

                # 计算边界框坐标
                img_height, img_width, _ = image.shape
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                # 绘制边界框和类别标签
                color = colors[class_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    classes[class_id],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # 保存带框的图像
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

print("处理完成！")

import os
import shutil
import xml.etree.ElementTree as ET
from config import BASE_DIR


class DataProcessor:
    def __init__(self, classes=[]) -> None:

        # 类别列表
        self.classes = classes

        # 文件夹路径
        self.origin_data_dir = os.path.join(BASE_DIR, "app", "origin_data")
        self.annotations_dir = os.path.join(self.origin_data_dir, "Annotations")
        self.jpeg_images_dir = os.path.join(self.origin_data_dir, "JPEGImages")
        self.image_sets_dir = os.path.join(self.origin_data_dir, "ImageSets/Main")

        # YOLO数据集输出文件夹
        self.yolo_base_dir = os.path.join(BASE_DIR, "app", "datasets")
        self.yolo_images_dir = os.path.join(self.yolo_base_dir, "images")
        self.yolo_labels_dir = os.path.join(self.yolo_base_dir, "labels")

    # 将Pascal VOC标注转换为YOLO格式
    def convert_annotation(self, xml_file, output_txt_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        with open(output_txt_path, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in self.classes:
                    continue  # 如果类别不在列表中，跳过此对象

                class_id = self.classes.index(class_name)

                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # 计算YOLO格式的中心点和宽高
                x_center = (xmin + xmax) / 2.0 / image_width
                y_center = (ymin + ymax) / 2.0 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                # 写入YOLO格式：<class_id> <x_center> <y_center> <width> <height>
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

    # 函数：复制图片并转换标签
    def process_dataset(self, split_name):
        split_file = os.path.join(self.image_sets_dir, f"{split_name}.txt")
        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f.readlines()]

        for image_id in image_ids:
            # 原始图片路径和标签路径
            image_path = os.path.join(self.jpeg_images_dir, f"{image_id}.jpg")
            xml_path = os.path.join(self.annotations_dir, f"{image_id}.xml")

            # 新的图片路径和标签路径
            output_image_dir = os.path.join(self.yolo_images_dir, split_name)
            output_label_dir = os.path.join(self.yolo_labels_dir, split_name)

            # 确保目录存在，递归创建文件夹
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            output_image_path = os.path.join(output_image_dir, f"{image_id}.jpg")
            output_txt_path = os.path.join(output_label_dir, f"{image_id}.txt")

            # 复制图片到新位置
            shutil.copy(image_path, output_image_path)

            # 转换XML标签为YOLO格式
            self.convert_annotation(xml_path, output_txt_path)


if __name__ == "__main__":
    print("开始转换数据集...")
    # 类别名称列表，与原始数据集中的类别保持一致
    classes = [
        "A220",
        "A320/321",
        "A330",
        "ARJ21",
        "Boeing737",
        "Boeing787",
        "other",
    ]
    data_processor = DataProcessor(classes=classes)
    data_processor.process_dataset("train")
    data_processor.process_dataset("val")
    data_processor.process_dataset("test")
    print("数据集转换完成！")

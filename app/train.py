import argparse
from ultralytics import YOLO
import os
from config import BASE_DIR, TRAIN_PT


def train(
    epochs: int = 500, patience: int = 100, GPU_devices: list = [0]
):
    """
    训练YOLO模型
    :param epochs: 训练周期，默认为500
    :param patience: 多少个周期没有提升就停止训练，默认为100
    :param GPU_devices: 使用的GPU设备，一个列表，如[0,1,2,3]，可以通过nvdia-smi进行设备列表查看，默认为[0]
    :return: 训练好的模型
    """

    # 设置GPU设备
    cuda_visible_devices = ",".join(map(str, GPU_devices))

    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # 数据集配置文件路径
    data_yaml = os.path.join(BASE_DIR, "app", "data.yaml")

    # 加载YOLO预训练模型
    model = YOLO(TRAIN_PT)

    # 训练模型
    model.train(
        data=data_yaml,  # 训练数据集配置文件
        epochs=epochs,  # 训练周期
        patience=patience,  # 多少个周期没有提升就停止训练
        imgsz=640,  # 图像大小
        batch=0.9,  # 批处理大小，自动占用显存90%
        device=GPU_devices,  # 使用的GPU设备
        val=False,
    )

    return model


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Train YOLO model with custom parameters."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default is 100.",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=100,
        help="Patience for early stopping. Default is 100.",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        default="0",
        help="Comma-separated list of GPU devices (e.g., '0,1,2'). Default is '0'.",
    )

    args = parser.parse_args()

    # 将 GPU 列表从字符串转换为整数列表
    gpu_devices = list(map(int, args.gpu.split(",")))

    # 调用训练函数
    train(
        epochs=args.epochs,
        patience=args.patience,
        GPU_devices=gpu_devices,
    )

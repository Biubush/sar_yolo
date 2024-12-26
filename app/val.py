import argparse
from ultralytics import YOLO
import os
from config import BEST_PT, BASE_DIR


def validate(batch: int = 16, GPU_devices: list = [0]):
    """
    验证YOLO模型
    :param model_path: 模型文件路径
    :param data_path: 验证数据集配置文件路径
    :param batch: 批处理大小，默认为16
    :param GPU_devices: 使用的GPU设备，一个列表，如[0,1,2,3]，默认为[0]
    """
    data_path = os.path.join(BASE_DIR, "app", "data.yaml")

    # 设置GPU设备
    cuda_visible_devices = ",".join(map(str, GPU_devices))

    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # 加载模型
    model = YOLO(BEST_PT)

    # 验证模型
    results = model.val(
        data=data_path,  # 验证数据集配置文件路径
        batch=batch,  # 批处理大小
        device=GPU_devices,  # 使用的GPU设备
    )

    print("验证完成，请查看输出文件夹下的图片和数据")


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Validate YOLO model with custom parameters."
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=16,
        help="Batch size for validation. Default is 16.",
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

    # 调用验证函数
    validate(batch=args.batch, GPU_devices=gpu_devices)

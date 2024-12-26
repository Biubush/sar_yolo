from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from config import BASE_DIR,BEST_PT

def predict_test_img(image_file):
    # 训练出的模型文件路径
    model_name = BEST_PT

    # 加载YOLO模型
    model = YOLO(model_name)

    # 将文件对象转换为图像
    image = Image.open(image_file)

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        # 将图像保存为临时JPEG文件
        image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name  # 获取临时文件路径

    try:
        # 测试模型
        model.predict(
            source=temp_file_path,  # 使用临时文件路径
            save=True,              # 保存带有标签和概率的测试结果图片
            conf=0.25,              # 置信度阈值，默认0.25
            name='result',       # 保存的测试结果图片的名称
            exist_ok=True           # 如果存在同名文件，是否覆盖
        )

        print(f"{'-'*10}预测完成{'-'*10}")
        # 删除临时文件
        os.remove(temp_file_path)
        result_file_name=temp_file.name.split('\\')[-1]
        output_path = os.path.join(BASE_DIR,'runs', 'detect', 'result', result_file_name)
        result_image = Image.open(output_path)

    except Exception as e:
        print(f"{'-'*10}预测失败{'-'*10}")
        print(e)
        result_image = None
    return result_image

if __name__ == '__main__':
    # 假设你有一个文件对象，可以直接传入
    with open(r'E:\documents\研究生\程序设计课程\代码\app\datasets\images\test\0000011.jpg', 'rb') as f:
        predict_test_img(f)

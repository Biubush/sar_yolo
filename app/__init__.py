from flask import Flask,render_template, request, jsonify
import base64
from PIL import Image
import io
from .predict import predict_test_img
import os

def create_app():
    app = Flask(__name__)

    # 应用配置
    app.config['SECRET_KEY'] = 'xb12986vc21n3e7894'

    return app

app = create_app()

import io
import base64
from PIL import Image
import tempfile
from config import BASE_DIR

def process_image(file):
    """处理图像的示例函数，返回两种不同的图像效果。"""
    # 获取上传的图像文件名
    filename = file.filename

    # 构建路径
    origin_image_path = os.path.join(BASE_DIR, 'app', 'origin_test_wirth_labels', filename)
    
    # 读取上传的图像
    img = Image.open(file)

    # 创建临时文件来保存图像，以便传递给predict函数
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        img.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name

    try:
        img1 = predict_test_img(temp_file_path)  # 使用YOLO模型预测
    finally:
        # 清理临时文件
        os.remove(temp_file_path)

    # 读取指定路径的图像
    img2 = Image.open(origin_image_path)

    # 转换为 Base64 编码
    buffer1 = io.BytesIO()
    img1.save(buffer1, format="PNG")
    img1_base64 = base64.b64encode(buffer1.getvalue()).decode('utf-8')

    buffer2 = io.BytesIO()
    img2.save(buffer2, format="PNG")
    img2_base64 = base64.b64encode(buffer2.getvalue()).decode('utf-8')

    return img1_base64, img2_base64

@app.route('/recieve_picture', methods=['POST'])
def receive_picture():
    file = request.files['image']
    img1_base64, img2_base64 = process_image(file)
    return jsonify(image1=img1_base64, image2=img2_base64)

@app.route('/')
def index():
    return render_template('index.html')
import os

# 当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 预训练模型完整路径
TRAIN_PT = os.path.join(BASE_DIR, 'app','yolov5su.pt')

# 你可以用我训练完成的最佳权重文件进行验收
BEST_PT = os.path.join(BASE_DIR, 'app','best.pt')

# # 或者使用自行训练后获得的最佳权重，这里是不对代码做修改时的默认路径，使用自行训练时请注释上面的BEST_PT行并解除下一行的注释（增删#字符）
# BEST_PT = os.path.join(BASE_DIR,'runs','detect','train','weights','best.pt')
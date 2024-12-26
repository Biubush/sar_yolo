from app import train,val

if __name__=="__main__":
    # 训练模型
    train.train()
    # 验证模型
    val.validate()
from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import random

def prepare_dataset():
    """准备数据集"""
    print("正在准备数据集...")
    
    # 创建数据分割
    image_dir = Path("datasets/images")
    label_dir = Path("datasets/labels")
    
    # 获取所有图片
    images = list(image_dir.glob("*.jpg")) + \
             list(image_dir.glob("*.png")) + \
             list(image_dir.glob("*.jpeg"))
    
    # 随机打乱
    random.shuffle(images)
    
    # 分割训练集和验证集（80%训练，20%验证）
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"训练集: {len(train_images)} 张图片")
    print(f"验证集: {len(val_images)} 张图片")
    
    # 创建目录
    (Path("datasets") / "train" / "images").mkdir(parents=True, exist_ok=True)
    (Path("datasets") / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (Path("datasets") / "val" / "images").mkdir(parents=True, exist_ok=True)
    (Path("datasets") / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # 复制文件到训练集
    for img_path in train_images:
        # 复制图片
        shutil.copy(img_path, Path("datasets/train/images") / img_path.name)
        
        # 复制标签（如果存在）
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, Path("datasets/train/labels") / label_path.name)
    
    # 复制文件到验证集
    for img_path in val_images:
        # 复制图片
        shutil.copy(img_path, Path("datasets/val/images") / img_path.name)
        
        # 复制标签（如果存在）
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, Path("datasets/val/labels") / label_path.name)
    
    # 创建YOLO数据配置文件
    yaml_content = """path: datasets/
train: train/images
val: val/images

nc: 1  # 类别数
names: ['first_aid_kit']  # 类别名称
"""
    
    with open("datasets/data.yaml", "w") as f:
        f.write(yaml_content)
    
    print("数据集准备完成！")

def train_model():
    """训练模型"""
    print("开始训练YOLO模型...")
    
    # 加载预训练模型（自动下载）
    model = YOLO("yolov8n.pt")  # 使用最小的模型
    
    # 训练模型
    model.train(
        data="datasets/data.yaml",  # 数据配置文件
        epochs=50,                  # 训练轮数（小数据集可以少一些）
        imgsz=640,                  # 图片大小
        batch=4,                    # 批大小
        device="cpu",               # 使用CPU（如果有GPU可以改成0）
        name="firstaid_model",      # 模型名称
        save=True,                  # 保存模型
    )
    
    print("训练完成！")
    
    # 测试一下模型
    test_model()

def test_model():
    """测试模型"""
    print("正在测试模型...")
    
    # 加载训练好的模型
    model = YOLO("runs/detect/firstaid_model/weights/best.pt")
    
    # 找一个测试图片
    test_images = list(Path("datasets/val/images").glob("*"))
    
    if test_images:
        # 测试第一张图片
        results = model(test_images[0], save=True, conf=0.25)
        
        print(f"测试图片: {test_images[0].name}")
        if results[0].boxes:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                print(f"检测到急救箱，置信度: {conf:.2f}")
        else:
            print("未检测到急救箱")
        
        print("测试结果保存在: runs/detect/predict")
    else:
        print("没有测试图片")

if __name__ == "__main__":
    print("=== 一键训练程序 ===")
    
    # 1. 准备数据
    prepare_dataset()
    
    # 2. 训练模型
    train_model()
    
    print("全部完成！")
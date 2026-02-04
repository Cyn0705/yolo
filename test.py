import cv2
import glob
from ultralytics import YOLO
from pathlib import Path

def test_new_images():
    """测试新图片"""
    print("=== 急救箱检测测试 ===")
    
    # 创建测试文件夹
    Path("test_images").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    print("请将要测试的图片放入 'test_images' 文件夹")
    input("按回车键开始检测...")
    
    # 获取测试图片
    test_files = glob.glob("test_images/*.jpg") + \
                glob.glob("test_images/*.png") + \
                glob.glob("test_images/*.jpeg")
    
    if len(test_files) == 0:
        print("在 test_images 文件夹中没有找到图片！")
        print("请放入一些测试图片，然后重新运行。")
        return
    
    # 加载模型
    print("加载模型中...")
    model = YOLO("runs/detect/firstaid_model/weights/best.pt")
    
    # 逐个检测图片
    for img_path in test_files:
        print(f"\n处理: {Path(img_path).name}")
        
        # 检测
        results = model(img_path, conf=0.25)  # 置信度阈值0.25
        
        # 显示结果
        if results[0].boxes:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                print(f"  ✓ 检测到急救箱，置信度: {conf:.3f}")
            
            # 保存带标注的图片
            annotated = results[0].plot()
            output_path = f"results/{Path(img_path).name}"
            cv2.imwrite(output_path, annotated)
            print(f"  结果保存到: {output_path}")
            
            # 显示图片
            cv2.imshow("检测结果", annotated)
            cv2.waitKey(3000)  # 显示3秒
        else:
            print("  ✗ 未检测到急救箱")
    
    cv2.destroyAllWindows()
    
    # 实时摄像头检测
    print("\n是否要尝试实时摄像头检测？ (y/n)")
    if input().lower() == 'y':
        realtime_detection(model)

def realtime_detection(model):
    """实时摄像头检测"""
    print("\n启动摄像头检测...")
    print("按 'q' 退出")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        results = model(frame, conf=0.25)
        
        # 绘制结果
        if results[0].boxes:
            annotated = results[0].plot()
        else:
            annotated = frame
        
        # 显示
        cv2.imshow("实时急救箱检测", annotated)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_new_images()
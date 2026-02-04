import cv2
import os
import glob
import json
from pathlib import Path

class SimpleLabeler:
    def __init__(self):
        # 设置文件夹
        self.image_dir = "datasets/images"  # 图片文件夹
        self.label_dir = "datasets/labels"  # 标签文件夹
        self.class_name = "first_aid_kit"   # 类别名称
        
        # 创建文件夹
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        Path(self.label_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片
        self.images = glob.glob(f"{self.image_dir}/*.jpg") + \
                     glob.glob(f"{self.image_dir}/*.png") + \
                     glob.glob(f"{self.image_dir}/*.jpeg")
        
        if len(self.images) == 0:
            print(f"错误：在 {self.image_dir} 中没有找到图片！")
            print("请将你的157张图片复制到该文件夹中")
            exit()
            
        print(f"找到 {len(self.images)} 张图片")
        
        # 当前状态
        self.current_idx = 0
        self.current_boxes = []  # 存储当前图片的所有框
        self.current_box = []    # 正在绘制的框 [x1, y1, x2, y2]
        self.drawing = False
    
    def mouse_event(self, event, x, y, flags, param):
        """鼠标事件处理"""
        image = param[0]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始画框
            self.drawing = True
            self.current_box = [x, y, x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 更新框的大小
            if self.drawing:
                self.current_box[2] = x
                self.current_box[3] = y
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 结束画框
            self.drawing = False
            self.current_box[2] = x
            self.current_box[3] = y
            
            # 确保坐标正确（左上到右下）
            x1 = min(self.current_box[0], self.current_box[2])
            y1 = min(self.current_box[1], self.current_box[3])
            x2 = max(self.current_box[0], self.current_box[2])
            y2 = max(self.current_box[1], self.current_box[3])
            
            # 过滤掉太小的框
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.current_boxes.append([x1, y1, x2, y2])
                self.current_box = []
    
    def draw_boxes(self, image):
        """在图片上绘制所有框"""
        img_copy = image.copy()
        
        # 绘制已完成的框
        for box in self.current_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, self.class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制正在画的框
        if self.drawing and len(self.current_box) == 4:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 显示信息
        info = f"图片: {self.current_idx+1}/{len(self.images)} | 框数量: {len(self.current_boxes)}"
        cv2.putText(img_copy, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return img_copy
    
    def save_label(self, image_path):
        """保存标签为YOLO格式"""
        if len(self.current_boxes) == 0:
            print("没有标注框，跳过保存")
            return False
            
        # 获取图片尺寸
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 创建标签文件路径
        filename = Path(image_path).stem
        label_path = f"{self.label_dir}/{filename}.txt"
        
        # 转换为YOLO格式并保存
        with open(label_path, 'w') as f:
            for box in self.current_boxes:
                x1, y1, x2, y2 = box
                
                # 转换为YOLO格式 (center_x, center_y, width, height)
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                # YOLO格式: class_id center_x center_y width height
                f.write(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        print(f"已保存: {label_path}")
        return True
    
    def load_label(self, image_path):
        """加载已有的标签"""
        filename = Path(image_path).stem
        label_path = f"{self.label_dir}/{filename}.txt"
        
        self.current_boxes = []
        
        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # 从YOLO格式转换回来
                        _, cx, cy, bw, bh = map(float, parts)
                        
                        # 转换为像素坐标
                        center_x = cx * width
                        center_y = cy * height
                        box_width = bw * width
                        box_height = bh * height
                        
                        x1 = int(center_x - box_width/2)
                        y1 = int(center_y - box_height/2)
                        x2 = int(center_x + box_width/2)
                        y2 = int(center_y + box_height/2)
                        
                        self.current_boxes.append([x1, y1, x2, y2])
            
            print(f"已加载 {len(self.current_boxes)} 个标注框")
    
    def run(self):
        """运行标注工具"""
        if len(self.images) == 0:
            return
            
        cv2.namedWindow("急救箱标注工具", cv2.WINDOW_NORMAL)
        
        while self.current_idx < len(self.images):
            # 加载图片
            image_path = self.images[self.current_idx]
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"无法读取图片: {image_path}")
                self.current_idx += 1
                continue
            
            # 设置鼠标回调
            cv2.setMouseCallback("急救箱标注工具", self.mouse_event, [image])
            
            # 加载已有标注
            self.load_label(image_path)
            
            print(f"\n当前图片: {Path(image_path).name}")
            print("使用说明:")
            print("  鼠标左键拖动: 绘制边界框")
            print("  's': 保存并进入下一张")
            print("  'n': 下一张图片（不保存）")
            print("  'p': 上一张图片")
            print("  'd': 删除最后一个框")
            print("  'q': 退出")
            
            while True:
                # 显示图片和框
                display = self.draw_boxes(image)
                cv2.imshow("急救箱标注工具", display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    if self.save_label(image_path):
                        self.current_idx += 1
                        self.current_boxes = []
                    break
                    
                elif key == ord('n'):  # 下一张（不保存）
                    self.current_idx += 1
                    self.current_boxes = []
                    break
                    
                elif key == ord('p'):  # 上一张
                    self.current_idx = max(0, self.current_idx - 1)
                    self.current_boxes = []
                    break
                    
                elif key == ord('d'):  # 删除最后一个框
                    if self.current_boxes:
                        self.current_boxes.pop()
                        print(f"删除一个框，还剩 {len(self.current_boxes)} 个")
                    else:
                        print("没有框可以删除")
                        
                elif key == ord('q'):  # 退出
                    cv2.destroyAllWindows()
                    print("标注已保存")
                    return
        
        cv2.destroyAllWindows()
        print("所有图片标注完成！")

# 主程序
if __name__ == "__main__":
    print("=== 急救箱标注工具 ===")
    print("")
    print("按任意键开始标注...")
    input()
    
    labeler = SimpleLabeler()
    labeler.run()
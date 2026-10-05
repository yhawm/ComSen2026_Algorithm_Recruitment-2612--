from ultralytics import YOLO
import os

# 确保输出文件夹存在
os.makedirs('../output/yolo', exist_ok=True)

# 1. 加载官方预训练权重（会自动下载 yolov8n.pt）
model = YOLO('yolov8n.pt') 
print("成功加载官方预训练模型 yolov8n.pt")

# 2. 指向 example 文件夹
img_dir = 'example' 
if not os.path.exists(img_dir):
    print(f"找不到文件夹 {img_dir}，请确认路径。")
    exit()

# 3. 遍历文件夹里的图片
for filename in os.listdir(img_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(img_dir, filename)
        print(f"正在推理: {filename}...")
        
        # 4. 进行推理
        results = model(img_path)
        
        # 5. 保存带框的图片
        save_path = f'../output/yolo/official_{filename}'
        results[0].save(save_path) # 使用 ultralytics 自带的方法保存
        print(f"已保存至: {save_path}")

print("第2点完成！请去 output/yolo/ 文件夹查看官方模型的推理结果。")
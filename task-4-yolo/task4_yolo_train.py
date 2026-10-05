from ultralytics import YOLO
import os

# ================= 必须加这行：Windows 多进程保护 =================
if __name__ == '__main__':
    os.makedirs('../output/yolo', exist_ok=True)

    # 加载无人机预训练权重
    model = YOLO('drone.pt') 
    print("成功加载 drone.pt，开始微调训练...")

    # 开始训练
    results = model.train(
        data='../drone_dataset/data.yaml',  
        epochs=50,                         
        imgsz=640,
        batch=16,                          # 显存 8GB，如果报 OOM 错误就改成 8 或 4
        device=0,                          # 使用 GPU 0
        workers=0,                         # 【关键修改】改为 0，彻底避免 Windows 多进程报错
        project='../output/yolo',          
        name='drone_training',
        exist_ok=True
    )

    print("训练完成！")
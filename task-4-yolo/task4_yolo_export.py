from ultralytics import YOLO
import os

# ================= 修改模型读取路径 =================
model_path = 'runs/output/yolo/drone_training/weights/best.pt'
if not os.path.exists(model_path):
    model_path = 'F:/py/ComSen2026_Algorithm_Recruitment/task-4/runs/output/yolo/drone_training/weights/best.pt'

model = YOLO(model_path)

print("===== 开始模型导出与量化 =====")

# 1. 导出 ONNX (FP32)
try:
    print("正在导出 ONNX (FP32)...")
    model.export(format='onnx', imgsz=640, half=False)
    print("ONNX (FP32) 导出成功！")
except Exception as e:
    print(f"ONNX 导出失败: {e}")

# 2. 导出 ONNX (FP16)
try:
    print("正在导出 ONNX (FP16)...")
    model.export(format='onnx', imgsz=640, half=True)
    print("ONNX (FP16) 导出成功！")
except Exception as e:
    print(f"ONNX FP16 导出失败: {e}")

# 3. 导出 TensorRT (FP16)
try:
    print("正在导出 TensorRT Engine (FP16)...")
    model.export(format='engine', imgsz=640, half=True, device=0)
    print("TensorRT (FP16) 导出成功！")
except Exception as e:
    print(f"TensorRT 导出失败 (可能未安装 TensorRT 环境，可忽略): {e}")

# 4. 导出 TensorRT (INT8，需校准数据)
try:
    print("正在导出 TensorRT Engine (INT8)...")
    model.export(format='engine', imgsz=640, int8=True, data='../drone_dataset/data.yaml', device=0)
    print("TensorRT (INT8) 导出成功！")
except Exception as e:
    print(f"TensorRT INT8 导出失败: {e}")

print("\n===== 导出完成，请去 runs/output/yolo/drone_training/weights/ 查看生成的文件 =====")
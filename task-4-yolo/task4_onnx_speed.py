import onnxruntime as ort
import numpy as np
import time
import os

# 1. 模型路径（请根据你实际的路径修改）
onnx_path = 'runs/output/yolo/drone_training/weights/best.onnx'
if not os.path.exists(onnx_path):
    print(f"找不到 ONNX 模型: {onnx_path}")
    exit()

# 2. 模型大小
size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"===== 模型大小 =====\nbest.onnx 大小: {size_mb:.2f} MB")

# 3. 加载 ONNX 模型（优先使用 GPU）
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"加载成功，当前使用设备: {session.get_providers()}")
except Exception as e:
    print(f"加载失败: {e}")
    exit()

# 4. 模拟输入 (YOLO 输入尺寸通常为 1x3x640x640)
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 5. 预热
print("预热中...")
for _ in range(10):
    session.run(None, {input_name: dummy_input})

# 6. 正式测试 (循环 100 次)
print("开始速度测试...")
start_time = time.time()
for _ in range(100):
    session.run(None, {input_name: dummy_input})
end_time = time.time()

# 7. 计算 FPS
avg_time = (end_time - start_time) / 100 * 1000 # 毫秒
fps = 1000 / avg_time

print(f"\n===== ONNX 推理速度分析 =====")
print(f"平均推理时间: {avg_time:.2f} ms")
print(f"推理速度: {fps:.2f} FPS")
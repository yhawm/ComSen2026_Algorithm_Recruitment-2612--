from ultralytics import YOLO
import cv2
import os
import time

os.makedirs('../output/yolo', exist_ok=True)

# ================= 修改模型读取路径 =================
model_path = 'runs/output/yolo/drone_training/weights/best.pt'
if not os.path.exists(model_path):
    model_path = 'F:/py/ComSen2026_Algorithm_Recruitment/task-4/runs/output/yolo/drone_training/weights/best.pt'

model = YOLO(model_path)

# ================= 模型大小分析 =================
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"===== 模型大小分析 =====")
print(f"best.pt 模型大小: {model_size_mb:.2f} MB")

# ================= 视频推理 =================
video_path = 'example/drone_video.mp4'  # 确保视频文件在这个路径下
if not os.path.exists(video_path):
    print(f"找不到视频 {video_path}，请准备好视频文件")
    exit()

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('../output/yolo/drone_result.mp4', fourcc, fps, (w, h))

print("开始视频推理...")
frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    # 计算推理时间
    start = time.time()
    results = model(frame, verbose=False)
    end = time.time()
    total_time += (end - start)
    
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    
    # 实时显示（按 q 退出，但建议让它跑完）
    cv2.imshow('YOLO Drone Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

# ================= 指标分析：推理速度 =================
avg_fps = frame_count / total_time
print(f"\n===== 推理速度分析 =====")
print(f"共处理 {frame_count} 帧")
print(f"平均推理速度: {avg_fps:.2f} FPS")
print(f"视频已保存至 ../output/yolo/drone_result.mp4")
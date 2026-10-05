import cv2
import numpy as np
import os

def track_video(video_path, output_path, lower, upper):
    if not os.path.exists(video_path):
        print(f"错误：找不到视频文件 {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    else: fps = int(fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出 MP4 格式
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        print("警告：avc1 不可用，降级为 mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ====== 追踪器初始化 ======
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        tracker = cv2.TrackerCSRT_create()

    ret, first_frame = cap.read()
    if not ret: return

    # ====== 使用传入的 HSV 范围进行检测 ======
    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(largest)
        print(f"[{video_path}] 自动检测到小球，位置: {bbox}")
    else:
        print(f"[{video_path}] 颜色检测失败，请手动框选小球（按回车确认）")
        bbox = cv2.selectROI("Select Ball", first_frame, False)
        cv2.destroyAllWindows()
        if bbox == (0,0,0,0): return

    tracker.init(first_frame, bbox)
    lost_count = 0
    MAX_LOST = 30  # 允许短暂遮挡 30 帧

    print(f"开始处理 {video_path}... (千万不要按 q 提前退出！)")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        success, box = tracker.update(frame)

        if success:
            lost_count = 0
            x, y, bw, bh = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            lost_count += 1
            cv2.putText(frame, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if lost_count > MAX_LOST:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    new_bbox = cv2.boundingRect(largest)
                    
                    try:
                        tracker = cv2.legacy.TrackerCSRT_create()
                    except AttributeError:
                        tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, new_bbox)
                    lost_count = 0
                    print(f"帧 {frame_count}: 成功找回小球！")

        out.write(frame)

    cap.release()
    out.release()
    print(f"{video_path} 追踪完成，共处理 {frame_count} 帧。结果保存至 {output_path}")

# ================== 核心修改：分别定义两个视频的 HSV 范围 ==================

# 1. example3.mp4 的颜色范围（请替换为你测出来的数值）
# 例如：橙色小球
lower_3 = np.array([0, 100, 100])
upper_3 = np.array([10, 255, 255])

# 2. example4.mp4 的颜色范围（请替换为你测出来的数值）
# 例如：绿色小球
lower_4 = np.array([80, 10, 100])
upper_4 = np.array([120, 50, 200])

# 分别传入对应的颜色范围
track_video('example3.mp4', '../output/example3_tracking.mp4', lower_3, upper_3)
track_video('example4.mp4', '../output/example4_tracking.mp4', lower_4, upper_4)

print("所有视频处理完毕！")
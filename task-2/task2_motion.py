import cv2
import numpy as np
import os

video_path = 'example2.mp4'

if not os.path.exists(video_path):
    print(f"错误：找不到文件 {video_path}！")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频。")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 30.0
else: fps = int(fps)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频分辨率 {w}x{h}, FPS: {fps}")

# ====== 保持 mp4 格式输出 ======
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out_no_bg = cv2.VideoWriter('../output/example2_no_bg.mp4', fourcc, fps, (w, h))
out_with_bg = cv2.VideoWriter('../output/example2_with_bg.mp4', fourcc, fps, (w, h))

if not out_no_bg.isOpened() or not out_with_bg.isOpened():
    print("警告：avc1 编码不可用，自动降级为 mp4v 编码...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_no_bg = cv2.VideoWriter('../output/example2_no_bg.mp4', fourcc, fps, (w, h))
    out_with_bg = cv2.VideoWriter('../output/example2_with_bg.mp4', fourcc, fps, (w, h))

# ====== 参数已恢复为稳定版 ======
# 1. 恢复 varThreshold=16 (降低敏感度)
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

prev_frame = None
# 2. 恢复 5x5 核 (更好地平滑噪点)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("开始处理视频。请耐心等待自然结束，中途千万别按 q 强制退出！")
print("（强制退出会导致 mp4 文件损坏或无法快进）")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已处理 {frame_count} 帧...")

    # ========== 方法1：帧差法 ==========
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_binary = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_no_bg = frame.copy()
        for c in contours:
            # 3. 恢复面积阈值 > 50 (过滤掉小的噪点)
            if cv2.contourArea(c) > 50:  
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(frame_no_bg, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(frame_no_bg, "Frame Diff", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out_no_bg.write(frame_no_bg)
    else:
        frame_no_bg = frame.copy()
        out_no_bg.write(frame_no_bg)

    prev_frame = frame.copy()

    # ========== 方法2：MOG2 ==========
    fg_mask = mog2.apply(frame)
    # 4. 恢复阴影阈值 200 (过滤掉灰色阴影)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY) 
    
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_bg = frame.copy()
    for c in contours:
        # 3. 恢复面积阈值 > 50
        if cv2.contourArea(c) > 50:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(frame_with_bg, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
    cv2.putText(frame_with_bg, "MOG2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out_with_bg.write(frame_with_bg)

    # 为了保证 mp4 视频能正常快进，建议注释掉窗口显示
    # cv2.imshow('No BG Subtraction', frame_no_bg)
    # cv2.imshow('With MOG2', frame_with_bg)
    # if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
out_no_bg.release()
out_with_bg.release()
cv2.destroyAllWindows()
print(f"处理完成！共 {frame_count} 帧，视频已保存为 mp4 到 output 文件夹。")
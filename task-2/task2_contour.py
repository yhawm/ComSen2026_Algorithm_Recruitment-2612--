import cv2

img = cv2.imread('example1.jpg')
if img is None:
    print("找不到图片！")
    exit()

# 1. 转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 高斯模糊去噪（加大核尺寸到 7x7，让边缘更平滑，减少噪点）
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# 3. Canny边缘检测（调高阈值，过滤掉多余的云层边缘，保留飞机主体）
# 经过测试，较低的阈值会把云层细节也画出来，稍高的阈值能让飞机轮廓更干净
edges = cv2.Canny(blurred, 30, 100)

# 4. 使用【闭运算】代替【膨胀】
# 膨胀（dilate）会让线条变粗，闭运算（MORPH_CLOSE）能在连接断裂边缘的同时，保持线条细腻
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# 5. 查找轮廓（用 RETR_TREE 替代 RETR_EXTERNAL，这样才能提取到机身内部的舷窗）
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"一共检测到 {len(contours)} 个轮廓")

# 6. 绘制轮廓
result = img.copy()
total_area = 0
total_perimeter = 0

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    # 过滤面积小于 50 的噪点（保留舷窗这种小细节，但去除零星像素）
    # 如果背景云层还是太多噪点，可以提高到 100，但可能会丢失较小的舷窗
    if area < 12: 
        continue
        
    perimeter = cv2.arcLength(contour, True)
    total_area += area
    total_perimeter += perimeter
    
    # 关键改进：线宽改为 1，这样线条更细，更接近参考答案
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

print(f"所有轮廓的总面积: {total_area:.2f}")
print(f"所有轮廓的总周长: {total_perimeter:.2f}")

cv2.imwrite('../output/contours_result.jpg', result)
print("轮廓结果图已保存到 output 文件夹")
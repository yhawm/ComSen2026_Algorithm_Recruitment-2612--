import cv2
import numpy as np
from PIL import Image

# ========== 题1：打开显示图片，分离三通道 ==========
img = cv2.imread('example1.jpg')
if img is None:
    raise FileNotFoundError("找不到 example1.jpg，请确认文件在当前目录下")

# OpenCV 默认 BGR，转为 RGB 后用 Pillow 显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)
pil_img.show()

# 分离三个通道（OpenCV 顺序为 B、G、R）
b, g, r = cv2.split(img)

# 保存各通道图像（单通道灰度图）
cv2.imwrite('../output/channel_blue.jpg', b)
cv2.imwrite('../output/channel_green.jpg', g)
cv2.imwrite('../output/channel_red.jpg', r)

# 用 Pillow 分别显示
Image.fromarray(b).show(title='Blue Channel')
Image.fromarray(g).show(title='Green Channel')
Image.fromarray(r).show(title='Red Channel')

# ========== 题2：缩放、旋转、平移、翻转 ==========
h, w = img.shape[:2]

# 2.1 缩放：缩放到原图的 50%
resized = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('../output/resized.jpg', resized)

# 2.2 旋转：绕图像中心旋转 45 度
center = (w // 2, h // 2)
M_rotate = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M_rotate, (w, h))
cv2.imwrite('../output/rotated.jpg', rotated)

# 2.3 平移：向右下平移 (100, 50)
M_translate = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M_translate, (w, h))
cv2.imwrite('../output/translated.jpg', translated)

# 2.4 翻转：水平翻转
flipped_h = cv2.flip(img, 1)
cv2.imwrite('../output/flipped_horizontal.jpg', flipped_h)
# 垂直翻转
flipped_v = cv2.flip(img, 0)
cv2.imwrite('../output/flipped_vertical.jpg', flipped_v)

print("Task 2 题1、题2 完成，结果保存在 ../output/ 目录")
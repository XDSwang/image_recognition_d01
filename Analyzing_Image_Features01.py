import os
import sys

print (os.getcwd())
print (sys.argv[0])
print (os.path.split(os.path.realpath(__file__))[0])
src=os.path.split(os.path.realpath(__file__))[0]

import os
import sys
import cv2
import numpy as np

# 读取图像
image_path = src + "/m02.png"
image = cv2.imread(image_path)

# 转换图像为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # 使用高斯滤波平滑图像
# gray_image = cv2.GaussianBlur(gray_image, (3, 3), 2)

# 使用阈值进行分割
_, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# # 使用高斯滤波平滑图像
# thresholded_image = cv2.GaussianBlur(thresholded_image, (3, 3), 2)

# 执行膨胀操作
kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)
# 执行腐蚀操作
kernel = np.ones((3, 3), np.uint8)
eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

# eroded_image = cv2.GaussianBlur(eroded_image, (3, 3), 2)
# 查找图像中的所有边框
contours, _ = cv2.findContours(eroded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 绘制所有边框
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)


#执行腐蚀操作作
kernel = np.ones((2, 2), np.uint8)
contour_image = cv2.erode(contour_image, kernel, iterations=2)
# 执行膨胀操作
kernel = np.ones((2, 2), np.uint8)
contour_image = cv2.dilate(contour_image, kernel, iterations=3)
contour_image = cv2.GaussianBlur(contour_image, (3, 3), 2)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Thresholded Image", thresholded_image)
cv2.imshow("eroded_image", eroded_image)
cv2.imshow("Contour Image", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 目前不破坏原有结构的情况最后的处理结果了

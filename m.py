import cv2
import os
import numpy as np

def r_int(s: str):
    """将字符串分割成整数元组"""
    i = s.split("_")
    return int(i[0]), int(i[1])

def process_image(image_path):
    # 针对性处理当前功能都存在，可以参照修改处理差异的
    # 读取图像
    image = cv2.imread(image_path)   
    # image = cv2.bitwise_not(image)
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值进行分割
    _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    # 执行膨胀操作
    # kernel = np.ones((5, 5), np.uint8)
    # dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)
    # 执行腐蚀操作
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=3)
    # 查找图像中的所有边框
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制所有边框
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    #执行腐蚀操作作
    # kernel = np.ones((2, 2), np.uint8)
    # contour_image = cv2.erode(contour_image, kernel, iterations=2)
    # 执行膨胀操作
    # kernel = np.ones((2, 2), np.uint8)
    # contour_image = cv2.dilate(contour_image, kernel, iterations=3)
    return contour_image


def process_image01(image_path):
    # 针对性处理
    # 读取图像
    image = cv2.imread(image_path)   
    # 颜色取反/能去除内容不是白色的外面边框
    image = cv2.bitwise_not(image)
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用阈值进行分割
    _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=3)
    # 查找图像中的所有边框
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制所有边框
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image


def process_image02(image_path, area_threshold_large=20, area_threshold_small=60):
       
    image = cv2.imread(image_path)   
    # image = cv2.bitwise_not(image)
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值进行分割
    _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    # 执行膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=5)
    # 执行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=3)
    # 查找图像中的所有边框
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 可以对边框针对性删除增加处理。。。
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold_large :
            filtered_contours.append(contour)
    contours=filtered_contours
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area < area_threshold_small:
            filtered_contours.append(contour)

    contours=filtered_contours        
    # 绘制所有边框
    contour_image = np.zeros_like(image)
    # cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    #执行腐蚀操作作
    kernel = np.ones((3, 3), np.uint8)
    contour_image = cv2.erode(contour_image, kernel, iterations=1)
 
    return contour_image

def process_image03(image_path):
    # 目前没有处理出来
    # 针对性处理当前功能都存在，可以参照修改处理差异的
    # 读取图像
    image = cv2.imread(image_path)   
    # image = cv2.bitwise_not(image)
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值进行分割
    _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    # 执行膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=5)
    # 执行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=3)
    # 查找图像中的所有边框
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制所有边框
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
       # # 执行膨胀操作
    # kernel = np.ones((3, 3), np.uint8)
    # contour_image = cv2.dilate(contour_image, kernel, iterations=2)
    #执行腐蚀操作作
    kernel = np.ones((3, 3), np.uint8)
    contour_image = cv2.erode(contour_image, kernel, iterations=1)
 
    return contour_image

def opencv_match(img1, img2, threshold=0.3, threshold01=5):
    sift = cv2.SIFT_create()
    # img1 = cv2.imread(img_d, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(template_d, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # kp1, des1 = sift.detectAndCompute(np.float32(img1), None)
    # kp2, des2 = sift.detectAndCompute(np.float32(img2), None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # print(matches)
    result = []
    for match in matches:
        m1 = match[0]  # Get the first match
        if m1.distance < threshold * match[1].distance:
            pt1 = kp1[m1.queryIdx].pt
            pt2 = kp2[m1.trainIdx].pt
            a = str(round(pt1[0]) - round(pt2[0]))
            b = str(round(pt1[1]) - round(pt2[1]))
            result.append(a + "_" + b)
        
    if len(result) > 0:
        # 使用NumPy统计坐标差值出现的次数
        unique_elements, counts = np.unique(result, return_counts=True)
        
        # 找到出现次数大于等于阈值的坐标差值的索引
        valid_indices = np.where(counts >= threshold01)[0]
        
        if len(valid_indices) > 0:
            # 获取出现次数最多的坐标差值的数组
            most_common_values = unique_elements[valid_indices]
            print(most_common_values)
            return most_common_values.tolist()
    
    return None


def render_matched_boxes(image_path, template_path, matches):
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    h, w = template.shape[:2]

    for s in matches:
        point = r_int(s)
        cv2.rectangle(img, (point[0], point[1]), (point[0] + w, point[1] + h), (0, 0, 255), 2)

    return img


def m01():
    src = os.path.split(os.path.realpath(__file__))[0]
    a="./data/screenshot.png"
    b="./data/searchadd0507.png"
    img1=process_image01(src+a)
    img2=process_image01(src+b)
    # img1 = cv2.imread(src+a)
    # img2 = cv2.imread(src+b)
    cv2.imshow("Matching Result01", img1)
    cv2.imshow("Matching Result02", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    matches = opencv_match(img1, img2, 0.9)
    # print(matches)
    if matches is not None:
        result_image = render_matched_boxes(src+a, src+b, matches)
        cv2.namedWindow("Matching Result", 0)
        cv2.imshow("Matching Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No matches found.")
def m02():
    src = os.path.split(os.path.realpath(__file__))[0]
    a = "/tz/data/mmexport1693120102165.jpg"
    b = "/tz/data/m01.jpg"
    # 这里使用了针对性处理  目前可以做到百分百匹配，做不到百分百全部找到
    img1=process_image02(src+a)
    img2=process_image03(src+b)
    # img1 = cv2.imread(src+a)
    # img2 = cv2.imread(src+b)
    cv2.imshow("Matching Result01", img1)
    cv2.imshow("Matching Result02", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    matches = opencv_match(img1, img2, 0.9,1)
    # print(matches)
    if matches is not None:
        result_image = render_matched_boxes(src+a, src+b, matches)
        cv2.namedWindow("Matching Result", 0)
        cv2.imshow("Matching Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No matches found.")

def m():
    src = os.path.split(os.path.realpath(__file__))[0]
    a = "./data/m02.png"
    b = "./data/m02_01.png"
    # a="/screenshot.png"
    # b="/searchadd0508.png"
    img1=process_image(src+a)
    img2=process_image(src+b)

    cv2.imshow("Matching Result01", img1)
    cv2.imshow("Matching Result02", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    matches = opencv_match(img1, img2, 0.7)
    # print(matches)
    if matches is not None:
        result_image = render_matched_boxes(src+a, src+b, matches)
        cv2.namedWindow("Matching Result", 0)
        cv2.imshow("Matching Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No matches found.")

if __name__ == "__main__":
    # 目前可以做到百分百匹配，做不到百分百全部找到。
    m02()



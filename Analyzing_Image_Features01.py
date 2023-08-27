import cv2
import os
import numpy as np
import time
import sys

print(os.getcwd())
print(sys.argv[0])
print(os.path.split(os.path.realpath(__file__))[0])
src = os.path.split(os.path.realpath(__file__))[0]

def r_int(s: str):
    """将字符串分割成整数元组"""
    i = s.split("_")
    return int(i[0]), int(i[1])

def opencv_match(img_d, template_d, threshold=0.3):
    """
    使用SIFT特征匹配图像
    :param img_d: 大图像路径
    :param template_d: 模板图像路径
    :param threshold: 特征匹配阈值
    :return: 匹配结果的坐标差列表
    """
    sift = cv2.SIFT_create()
    img1 = cv2.imread(img_d, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(template_d, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    result = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < threshold * m2.distance:
            pt1 = kp1[m1.queryIdx].pt
            pt2 = kp2[m1.trainIdx].pt
            a = str(round(pt1[0]) - round(pt2[0]))
            b = str(round(pt1[1]) - round(pt2[1]))
            result.append(a + "_" + b)

    unique_results = []
    for id in result:
        if id not in unique_results:
            unique_results.append(id)

    return unique_results


def render_matched_boxes(image_path, template_path, matches):
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    h, w = template.shape[:2]

    for s in matches:
        point = r_int(s)
        cv2.rectangle(img, (point[0], point[1]), (point[0] + w, point[1] + h), (0, 0, 255), 2)

    return img

# if __name__ == "__main__":
#     src = os.path.split(os.path.realpath(__file__))[0]
#     matches = opencv_match(src+"/screenshot.png", src+"/searchadd0508.png", 0.3)
#     result_image = render_matched_boxes(src+"/screenshot.png", src+"/searchadd0508.png", matches)
    
#     cv2.namedWindow("Matching Result", 0)
#     cv2.imshow("Matching Result", result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def process_frame(frame, target_kp, target_des, similarity_threshold, output_folder):
    sift = cv2.SIFT_create()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(target_des, des, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    similarity = len(good_matches) / len(target_kp)
    if similarity >= similarity_threshold:
        # 获取匹配位置
        target_points = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算矩形框坐标
        x, y, w, h = cv2.boundingRect(target_points)

        # 创建匹配文件夹
        folder_name = os.path.join(output_folder, f"similarity_{int(similarity * 100)}")
        os.makedirs(folder_name, exist_ok=True)

        # 获取当前时间戳
        timestamp = int(time.time())
        output_filename = f"matched_frame_{timestamp}.jpg"
        output_path = os.path.join(folder_name, output_filename)
        cv2.imwrite(output_path, frame)

        # 绘制矩形框
        frame_with_rect = frame.copy()
        cv2.rectangle(frame_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 输出相似度信息
        print(f"Matched at similarity {similarity:.2f}, Output: {output_path}")

        return frame_with_rect
    else:
        return frame


def detect_and_save_matched_frames(target_image_path, video_path, output_folder, scale_factor=0.5, similarity_threshold=0.1):
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.resize(target_image, None, fx=scale_factor, fy=scale_factor)
    target_kp, target_des = cv2.SIFT_create().detectAndCompute(target_image, None)

    cap = cv2.VideoCapture(video_path)

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        processed_frame = process_frame(frame, target_kp, target_des, similarity_threshold, output_folder)

        cv2.imshow("Matching Result", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
target_image_path = src + '/te06.jpg'
video_path = src + '/Screenrecorder-2023-04-15-07-26-00-406.mp4'
output_folder = src + '/data'
scale_factor = 0.5
similarity_threshold = 0.1  # 根据需要调整阈值

detect_and_save_matched_frames(target_image_path, video_path, output_folder, scale_factor, similarity_threshold)

import cv2
import os
import numpy as np
import time
import sys

print(os.getcwd())
print(sys.argv[0])
print(os.path.split(os.path.realpath(__file__))[0])
src = os.path.split(os.path.realpath(__file__))[0]


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

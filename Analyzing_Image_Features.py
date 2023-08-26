import cv2
import os
import numpy as np
import time
import sys

print(os.getcwd())
print(sys.argv[0])
print(os.path.split(os.path.realpath(__file__))[0])
src = os.path.split(os.path.realpath(__file__))[0]


def detect_and_save_matched_frames(target_image_path, video_path, output_folder, scale_factor=0.5, similarity_threshold=0.7):
    target_image = cv2.imread(target_image_path)
    target_image = cv2.resize(target_image, None, fx=scale_factor, fy=scale_factor)
    target_height, target_width = target_image.shape[:2]

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    method = cv2.TM_CCOEFF_NORMED

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        result = cv2.matchTemplate(frame, target_image, method)
        locs = np.where(result >= similarity_threshold)

        for loc in zip(*locs[::-1]):
            top_left = loc
            bottom_right = (top_left[0] + target_width, top_left[1] + target_height)
            target_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 获取相似度（匹配结果值）
            similarity = result[loc[1], loc[0]]  # 注意行列顺序

            # 根据相似度创建输出文件夹
            folder_name = os.path.join(output_folder, f"similarity_{int(similarity * 100)}")
            os.makedirs(folder_name, exist_ok=True)

            # 获取当前时间戳
            timestamp = int(time.time())
            output_filename = f"matched_frame_{timestamp}.jpg"
            output_path = os.path.join(folder_name, output_filename)
            cv2.imwrite(output_path, target_roi)

            # 输出相似度信息（仅保留一位小数）
            similarity_str = str(round(similarity, 1))
            print(f"Matched at {loc}, Similarity: {similarity_str}")

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.namedWindow("Matching Result", 0)
        cv2.imshow("Matching Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
target_image_path = src + '/te06.jpg'
video_path = src + '/Screenrecorder-2023-04-15-07-26-00-406.mp4'
output_folder = src + '/data'
scale_factor = 0.5
similarity_threshold = 0.6

detect_and_save_matched_frames(target_image_path, video_path, output_folder, scale_factor, similarity_threshold)



import os
import sys

print (os.getcwd())
print (sys.argv[0])
print (os.path.split(os.path.realpath(__file__))[0])
src=os.path.split(os.path.realpath(__file__))[0]

import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow_model_optimization.sparsity import keras as sparsity


# 加载保存的模型目前较快的
save_model_path = os.path.join(src, 'path/to/save_model_image.h5')
loaded_model = tf.keras.models.load_model(save_model_path)

def load_and_predict(image):


    # 图像预处理
    image = cv2.resize(image, (54, 54))
    # image = image / 255.0  # 图像归一化
    image = np.expand_dims(image, axis=0)  # 扩展维度，适应模型输入

    # 使用模型进行预测
    predictions = loaded_model.predict(image)
    predicted_class = np.argmax(predictions[0])
    if predicted_class == 0:
        return True
    return False






def detect_and_save_matched_frames(target_image_path, video_path, output_folder, scale_factor=0.5, similarity_threshold=0.7):
    # 读取目标图像并进行缩小处理
    target_image = cv2.imread(target_image_path)
    target_image = cv2.resize(target_image, None, fx=scale_factor, fy=scale_factor)
    target_height, target_width = target_image.shape[:2]

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    # 缩小视频帧的尺寸
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)

    # 设置匹配方法
    method = cv2.TM_CCOEFF_NORMED

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小视频帧
        frame = cv2.resize(frame, (frame_width, frame_height))

        # 对视频帧进行模板匹配
        result = cv2.matchTemplate(frame, target_image, method)

        # 获取匹配结果中超过阈值的位置信息
        locs = np.where(result >= similarity_threshold)


        # 根据位置信息绘制目标边框并进行预测判断
        for loc in zip(*locs[::-1]):
            top_left = loc
            bottom_right = (top_left[0] + target_width, top_left[1] + target_height)
            target_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
       
            # 使用模型进行匹配
            is_similar = load_and_predict(target_roi)
            if is_similar:
                # 模型匹配成功
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                print(11)
                break
        # 显示结果
        cv2.namedWindow("Matching Result", 0)
        cv2.imshow("Matching Result", frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()






# 使用示例
target_image_path = os.path.join(src, 'image_1688792642141974.jpg')
video_path = os.path.join(src, 'Screenrecorder-2023-04-15-07-26-00-406.mp4')
output_folder = os.path.join(src, 'data')
scale_factor = 0.5
similarity_threshold = 0.8

detect_and_save_matched_frames(target_image_path, video_path, output_folder, scale_factor, similarity_threshold)
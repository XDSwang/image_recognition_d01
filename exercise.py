import os
import sys

print (os.getcwd())
print (sys.argv[0])
print (os.path.split(os.path.realpath(__file__))[0])
src=os.path.split(os.path.realpath(__file__))[0]


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 设置输入图像的大小
input_shape = (54, 54, 3)
num_classes = 2


def load_data(train_folder, validation_folder, target_size=(54, 54), batch_size=32):
    # 创建数据生成器
    data_generator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=[0.8, 1.2],  # 控制图像的放大和缩小范围
        brightness_range=[0.5, 1.5]  # 控制亮度调整范围
    )

    # 加载训练数据和验证数据
    train_data = data_generator.flow_from_directory(
        train_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_data = data_generator.flow_from_directory(
        validation_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, validation_data

# 使用示例
train_folder = os.path.join(src, 'path/to/train_folder')
validation_folder = os.path.join(src, 'path/to/validation_folder')
train_data, validation_data = load_data(train_folder, validation_folder)



model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=input_shape),
    Conv2D(32, (1, 1), activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# # 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(
    train_data,
    epochs=50,
    validation_data=validation_data
)


# # 保存模型
model.save(src+'/path/to/save_model_image.h5')




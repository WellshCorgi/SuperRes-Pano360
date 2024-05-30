import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

def load_srcnn_model():
    input_shape = (None, None, 1)
    inputs = Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(128, (9, 9), activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    outputs = tf.keras.layers.Conv2D(1, (5, 5), padding='same')(conv2)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.load_weights('./model/srcnn_weights.h5')  # 사전 학습된 가중치 파일의 경로
    return model

def enhance_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)
    y = y.astype(np.float32) / 255.0
    y = np.expand_dims(np.expand_dims(y, axis=0), axis=-1)
    y_enhanced = model.predict(y)
    y_enhanced = y_enhanced[0, :, :, 0]
    y_enhanced = (y_enhanced * 255.0).clip(0, 255).astype(np.uint8)
    enhanced_image = cv2.merge([y_enhanced, cr, cb])
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_YCrCb2BGR)
    return enhanced_image

def enhance_images_in_directory(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    srcnn_model = load_srcnn_model()
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is not None:
            enhanced_image = enhance_image(image, srcnn_model)
            output_path = os.path.join(output_folder, f"enhanced_{idx:04d}.jpg")
            cv2.imwrite(output_path, enhanced_image)
    
    print(f"Enhanced images saved to {output_folder}")

# 사용 예제
input_folder = './convert_mp4_to_jpg_winid_cam'
output_folder = './improve_winid_cam'
enhance_images_in_directory(input_folder, output_folder)

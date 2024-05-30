import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
import argparse

def extract_frames(video_path, output_folder, frame_interval=60, enhance=False):
    # 사전 학습된 SRCNN 모델을 로드하는 함수
    def load_srcnn_model():
        input_shape = (None, None, 1)
        inputs = Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv2D(128, (9, 9), activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        outputs = tf.keras.layers.Conv2D(1, (5, 5), padding='same')(conv2)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.load_weights('./model/srcnn_weights.h5')  # 사전 학습된 가중치 파일의 경로
        return model

    # SRCNN을 사용하여 이미지를 선명하게 만드는 함수
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

    # 비디오 파일을 읽어들입니다.
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 파일이 열리지 않는 경우 에러 메시지를 출력합니다.
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # 초당 프레임 수 (FPS)를 가져옵니다.
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 프레임 간격을 계산합니다.
    #frame_interval = int(fps * interval)
    
    
    # 출력 폴더가 없는 경우 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 프레임을 선명하게 해야 하는 경우 SRCNN 모델을 로드합니다.
    srcnn_model = load_srcnn_model() if enhance else None
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 지정한 간격에 따라 프레임을 저장합니다.
        if frame_count % frame_interval == 0:
            if enhance and srcnn_model:
                frame = enhance_image(frame, srcnn_model)
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1
    
    # 비디오 파일을 닫습니다.
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('-i', '--enhance', default=int(0), type=int , help='Whether to enhance frames using SRCNN / 1(True) or 0(False)')
    parser.add_argument('-f', '--frame_interval', type=int, default=60, help='Frame interval for extraction')
    args = parser.parse_args()
    
    video_path = './input.mp4'
    output_folder = './convert_mp4_to_jpg'
    extract_frames(video_path, output_folder, frame_interval=args.frame_interval, enhance=args.enhance)

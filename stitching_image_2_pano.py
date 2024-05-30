import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# 주어진 이미지 리스트의 각 이미지를 지정된 너비와 높이로 크기 조정.
# 너비 또는 높이 중 하나만 주어진 경우, 원본 이미지의 비율을 유지.
def resize_image(img, width=None, height=None):
    if width is None and height is None:
        return img
    else:
        h, w = img.shape[:2]
        if width is not None and height is not None:
            new_size = (width, height)
        elif width is not None:
            new_size = (width, int(h * (width / w)))
        else:
            new_size = (int(w * (height / h)), height)
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return resized_img

def stitch_images(images, output_path):
    # 이미지 스티칭을 위한 OpenCV 객체 생성
    stitcher = cv2.Stitcher_create()

    # 이미지 스티칭 수행
    status, stitched_img = stitcher.stitch(images)
    
    # 스티칭이 성공적으로 수행되었는지 확인
    if status == cv2.Stitcher_OK:
        print("이미지 스티칭 성공!")
        
        # marzipano에 적용하기 위함 - 스티칭된 이미지를 2:1 비율로 크기 조정 
        h, w = stitched_img.shape[:2]
        new_width = w
        new_height = int(w / 2)
        if new_height > h:
            new_height = h
            new_width = int(h * 2)
        resized_stitched_img = resize_image(stitched_img, new_width, new_height)
        
        # 최종 이미지를 저장
        cv2.imwrite(output_path, resized_stitched_img)
    else:
        print("이미지 스티칭 실패: ", status)
        # 실패한 경우 더 자세한 이유를 출력
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("더 많은 이미지가 필요합니다.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("호모그래피 추정 실패. 이미지가 너무 다릅니다.")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("카메라 파라미터 조정 실패.")
        else:
            print("알 수 없는 오류 발생")

def load_and_resize_image(filepath, width=None, height=None):
    img = cv2.imread(filepath)
    if img is not None:
        img = resize_image(img, width, height)
    return img

def main(input_folder, output_path, width=None, height=None):
    # 입력 사진의 확장자가 jpeg, jpg, png인 파일만 선택
    valid_extensions = (".jpeg", ".jpg", ".png")
    images = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in sorted(os.listdir(input_folder)):
            # 확장자가 유효한 경우
            if filename.lower().endswith(valid_extensions):  
                img_path = os.path.join(input_folder, filename)  # 이미지 파일의 전체 경로 생성
                futures.append(executor.submit(load_and_resize_image, img_path, width, height))

        for future in futures:
            img = future.result()
            if img is not None:
                images.append(img)  # 이미지를 리스트에 추가

    if len(images) == 0:
        print("이미지를 불러오지 못했습니다. 폴더 경로를 확인하세요.")
    else:
        # 이미지 스티칭 함수 호출
        stitch_images(images, output_path)

if __name__ == "__main__":
    input_folder = "./convert_mp4_to_jpg"
    output_path = "./stitched_result.jpg"
    main(input_folder, output_path, width=1500)

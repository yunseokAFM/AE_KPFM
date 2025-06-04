import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json
import warnings
from datetime import datetime
from typing import List, Tuple, Optional

# Detectron2 imports
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import matplotlib.pyplot as plt

setup_logger()
warnings.filterwarnings(action='ignore')

def get_cfg_cpu():
    """CPU용 Detectron2 설정 반환"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def get_object_dicts(img_dir: str) -> List[dict]:
    """
    간단한 데이터셋 로더 (실제 프로젝트에서는 적절히 수정 필요)
    """
    # 실제 구현에서는 어노테이션 파일을 읽어야 함
    return []

def find_latest_error_forward_file(directory: str) -> Optional[str]:
    """
    Error Signal_Forward가 포함된 가장 최근 파일 찾기
    
    Args:
        directory: 검색할 디렉토리 경로
        
    Returns:
        가장 최근 파일의 경로 또는 None
    """
    try:
        # 디렉토리 존재 확인
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist")
            return None
            
        # 패턴 매칭으로 파일 검색
        search_patterns = [
            os.path.join(directory, "*Error Signal_Forward*"),
            os.path.join(directory, "*Error*Signal*Forward*"),
            os.path.join(directory, "**", "*Error Signal_Forward*")  # 하위 폴더 포함
        ]
        
        files = []
        for pattern in search_patterns:
            files.extend(glob.glob(pattern, recursive=True))
        
        # 중복 제거
        files = list(set(files))
        
        if not files:
            print(f"No files containing 'Error Signal_Forward' found in {directory}")
            return None
        
        # 가장 최근 파일 찾기
        latest_file = max(files, key=os.path.getmtime)
        print(f"Found latest file: {latest_file}")
        return latest_file
        
    except Exception as e:
        print(f"Error searching for files: {e}")
        return None

def validate_image(image_path: str) -> bool:
    """이미지 파일 유효성 검사"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return False
    
    # 파일 크기 확인
    if os.path.getsize(image_path) == 0:
        print(f"Error: Image file {image_path} is empty")
        return False
        
    return True

def setup_detectron2_datasets():
    """Detectron2 데이터셋 등록"""
    for d in ["train", "val"]:
        dataset_name = f"object_{d}"
        
        # 기존 데이터셋 제거
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            
        # 새 데이터셋 등록
        try:
            DatasetCatalog.register(
                dataset_name, 
                lambda d=d: get_object_dicts(os.path.join("object", d))
            )
            MetadataCatalog.get(dataset_name).set(thing_classes=["object"])
            print(f"Successfully registered dataset: {dataset_name}")
        except Exception as e:
            print(f"Error registering dataset {dataset_name}: {e}")

def segmentation(test_directory: str, 
                model_path: str = None,
                score_threshold: float = 0.7,
                visualize: bool = True) -> Tuple[List[List[float]], List[float]]:
    """
    Detectron2를 사용한 이미지 세그멘테이션
    
    Args:
        test_directory: 테스트 이미지가 있는 디렉토리
        model_path: 훈련된 모델 경로 (None이면 기본 경로 사용)
        score_threshold: 검출 임계값
        visualize: 결과 시각화 여부
        
    Returns:
        Tuple[center_list, mask_sizes]: 중심 좌표와 마스크 크기 리스트
    """
    center_list = []
    mask_sizes = []
    
    try:
        # 데이터셋 설정
        setup_detectron2_datasets()
        
        # 모델 설정
        cfg = get_cfg_cpu()
        cfg.DATASETS.TRAIN = ("object_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 100
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        
        # 모델 가중치 설정
        if model_path is None:
            model_path = os.path.join("output", "model_final.pth")
            
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return center_list, mask_sizes
            
        cfg.MODEL.WEIGHTS = model_path
        predictor = DefaultPredictor(cfg)
        
        # 최신 이미지 파일 찾기
        latest_file = find_latest_error_forward_file(test_directory)
        if not latest_file:
            return center_list, mask_sizes
            
        # 이미지 유효성 검사
        if not validate_image(latest_file):
            return center_list, mask_sizes
        
        # 이미지 읽기
        im = cv2.imread(latest_file, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"Error: Could not read image {latest_file}")
            return center_list, mask_sizes
            
        print(f"Image shape: {im.shape}")
        
        # 예측 수행
        outputs = predictor(im)
        instances = outputs["instances"]
        
        if len(instances) == 0:
            print("No objects detected")
            return center_list, mask_sizes
            
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        masks = instances.pred_masks.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        print(f"Detected {len(instances)} objects")
        
        # 각 검출된 객체 처리
        for i, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            try:
                # 컨투어 찾기
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if len(contours) == 0:
                    print(f"No contours found for object {i}")
                    continue
                    
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 중심 좌표 계산
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    mask_size = cv2.contourArea(largest_contour)
                    
                    center_list.append([cX, cY, mask_size])
                    mask_sizes.append(mask_size)
                    
                    print(f"Object {i}: Center=({cX}, {cY}), Size={mask_size:.2f}, Score={score:.3f}")
                else:
                    print(f"Could not calculate moments for object {i}")
                    
            except Exception as e:
                print(f"Error processing object {i}: {e}")
                continue
        
        # 시각화
        if visualize and len(instances) > 0:
            try:
                # RGB로 변환 (BGR -> RGB)
                if len(im.shape) == 3:
                    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                else:
                    im_rgb = im
                    
                v = Visualizer(
                    im_rgb, 
                    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                    scale=0.5
                )
                out = v.draw_instance_predictions(instances.to("cpu"))
                
                plt.figure(figsize=(12, 8))
                plt.imshow(out.get_image())
                plt.title(f"Segmentation Results - {len(instances)} objects detected")
                plt.axis('off')
                plt.show()
                
            except Exception as e:
                print(f"Error during visualization: {e}")
        
        print(f"Segmentation completed successfully. Found {len(center_list)} valid objects.")
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        
    return center_list, mask_sizes

# 사용 예시
if __name__ == "__main__":
    test_dir = "Data/Approximate/241108_ZeroScan"
    centers, sizes = segmentation(test_dir)
    print(f"Results: {len(centers)} objects detected")
    for i, (center, size) in enumerate(zip(centers, sizes)):
        print(f"Object {i}: Center={center[:2]}, Size={size}")
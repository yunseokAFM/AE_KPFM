import os
import gc
import torch
import numpy as np
import cv2
import glob
import warnings
from typing import List, Tuple, Optional, Dict
from contextlib import contextmanager

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from readTiff import tiff2array, tiff_info

setup_logger()
warnings.filterwarnings(action='ignore')

class DeviceManager:    
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @staticmethod
    def get_memory_info():
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return None

@contextmanager
def memory_cleanup():
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class ConfigManager:    
    def get_detectron2_config(device: str = "auto", 
                            score_threshold: float = 0.7,
                            num_classes: int = 1) -> any:
        """Detectron2 start"""
        if device == "auto":
            device = DeviceManager.get_device()
            
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        if device == "cuda":
            cfg.DATALOADER.NUM_WORKERS = 0  
            cfg.SOLVER.IMS_PER_BATCH = 1   
        
        return cfg

class DatasetHandler:
    
    @staticmethod
    def get_object_dicts(img_dir: str) -> List[dict]:
        dataset_dicts = []
        
        try:
            tiff_files = glob.glob(os.path.join(img_dir, "*.tiff")) + \
                        glob.glob(os.path.join(img_dir, "*.tif"))
            
            if not tiff_files:
                print(f"Warning: No TIFF files found in {img_dir}")
                return dataset_dicts
                
            for tiff_file in tiff_files:
                try:
                    info_dict, _ = tiff_info(tiff_file)
                    header = info_dict['HEADER']
                    
                    record = {
                        "file_name": tiff_file,
                        "image_id": len(dataset_dicts),
                        "height": int(header.get('height', 256)),
                        "width": int(header.get('width', 256)),
                        "annotations": [] 
                    }
                    dataset_dicts.append(record)
                    
                except Exception as e:
                    print(f"Error processing {tiff_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing directory {img_dir}: {e}")
            
        return dataset_dicts
    
    def setup_datasets():
        for d in ["train", "val"]:
            dataset_name = f"object_{d}"
            
            # 기존 데이터셋 제거
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)
            
            try:
                DatasetCatalog.register(
                    dataset_name,
                    lambda d=d: DatasetHandler.get_object_dicts(os.path.join("object", d))
                )
                MetadataCatalog.get(dataset_name).set(thing_classes=["object"])
                print(f"Dataset registered: {dataset_name}")
            except Exception as e:
                print(f"Error registering {dataset_name}: {e}")

class FileManager:
    
    @staticmethod
    def find_latest_error_forward_file(directory: str) -> Optional[str]:
        """Error Signal_Forward 파일 검색 (개선된 버전)"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        patterns = [
            "*Error Signal_Forward*.tiff",
            "*Error Signal_Forward*.tif",
            "*Error*Signal*Forward*.tiff",
            "*Error*Signal*Forward*.tif"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
        
        if not files:
            return None
            
        return max(files, key=os.path.getmtime)
    
    @staticmethod
    def validate_file(file_path: str, min_size: int = 1024) -> bool:
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) < min_size:
            return False
        return True

class OptimizedSegmentation:    
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = DeviceManager.get_device() if device == "auto" else device
        self.predictor = None
        self.model_path = model_path or os.path.join("output", "model_final.pth")
        
        DatasetHandler.setup_datasets()
        
    def load_model(self, score_threshold: float = 0.7):
        try:
            if not FileManager.validate_file(self.model_path):
                print("Custom model not found, using pretrained model")
                cfg = ConfigManager.get_detectron2_config(self.device, score_threshold)
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            else:
                cfg = ConfigManager.get_detectron2_config(self.device, score_threshold)
                cfg.MODEL.WEIGHTS = self.model_path
            
            cfg.DATASETS.TRAIN = ("object_train",)
            self.predictor = DefaultPredictor(cfg)
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_tiff_image(self, tiff_path: str) -> Optional[np.ndarray]:
        
        try:
            # TIFF to array
            image_array = tiff2array(tiff_path)
            
            # Normalization
            if image_array.max() > 255:
                image_array = (image_array / image_array.max() * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
            
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            return image_array
            
        except Exception as e:
            print(f"Error loading TIFF {tiff_path}: {e}")
            return None
    
    def process_predictions(self, outputs) -> Tuple[List[List[float]], List[float]]:
        """예측 결과 처리"""
        center_list = []
        mask_sizes = []
        
        instances = outputs["instances"]
        if len(instances) == 0:
            return center_list, mask_sizes
        
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        masks = instances.pred_masks.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        for i, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            try:
                # Find contour
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    continue
                
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate center point
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    mask_size = cv2.contourArea(largest_contour)
                    
                    center_list.append([cX, cY, mask_size])
                    mask_sizes.append(mask_size)
                    
            except Exception as e:
                print(f"Error processing object {i}: {e}")
                continue
        
        return center_list, mask_sizes
    
    def segment_directory(self, test_directory: str, 
                         score_threshold: float = 0.7) -> Tuple[List[List[float]], List[float]]:
        
        with memory_cleanup():
            try:
                # Load model
                if self.predictor is None:
                    self.load_model(score_threshold)
                
                # Find latest approxiamte scan file
                latest_file = FileManager.find_latest_error_forward_file(test_directory)
                if not latest_file:
                    print("No suitable files found")
                    return [], []
                
                if not FileManager.validate_file(latest_file):
                    print(f"Invalid file: {latest_file}")
                    return [], []
                
                # Load image
                image = self.load_tiff_image(latest_file)
                if image is None:
                    return [], []
                
                print(f"Processing: {latest_file}, Shape: {image.shape}")
                
                outputs = self.predictor(image)
                
                center_list, mask_sizes = self.process_predictions(outputs)
                
                print(f"Detected {len(center_list)} objects")
                return center_list, mask_sizes
                
            except Exception as e:
                print(f"Segmentation error: {e}")
                return [], []


def segmentation(test_directory: str, 
                model_path: str = None,
                score_threshold: float = 0.7,
                visualize: bool = False) -> Tuple[List[List[float]], List[float]]:
    
    seg = OptimizedSegmentation(model_path)
    return seg.segment_directory(test_directory, score_threshold)

if __name__ == "__main__":
    device = DeviceManager.get_device()
    print(f"Using device: {device}")
    
    if device == "cuda":
        memory = DeviceManager.get_memory_info()
        print(f"GPU Memory: {memory / 1e9:.1f} GB")
    
    test_dir = "Data/Approximate/241108_ZeroScan"
    centers, sizes = segmentation(test_dir)
    print(f"Results: {len(centers)} objects detected")

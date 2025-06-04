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
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def get_object_dicts(img_dir: str) -> List[dict]:
    return []

def find_latest_error_forward_file(directory: str) -> Optional[str]:

    try:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist")
            return None
            

        search_patterns = [
            os.path.join(directory, "*Error Signal_Forward*"),
            os.path.join(directory, "*Error*Signal*Forward*"),
            os.path.join(directory, "**", "*Error Signal_Forward*")
        ]
        
        files = []
        for pattern in search_patterns:
            files.extend(glob.glob(pattern, recursive=True))

        files = list(set(files))
        
        if not files:
            print(f"No files containing 'Error Signal_Forward' found in {directory}")
            return None
        

        latest_file = max(files, key=os.path.getmtime)
        print(f"Found latest file: {latest_file}")
        return latest_file
        
    except Exception as e:
        print(f"Error searching for files: {e}")
        return None

def validate_image(image_path: str) -> bool:

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return False
    

    if os.path.getsize(image_path) == 0:
        print(f"Error: Image file {image_path} is empty")
        return False
        
    return True

def setup_detectron2_datasets():

    for d in ["train", "val"]:
        dataset_name = f"object_{d}"
        

        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            

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

    center_list = []
    mask_sizes = []
    
    try:

        setup_detectron2_datasets()
        

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
        

        if model_path is None:
            model_path = os.path.join("output", "model_final.pth")
            
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return center_list, mask_sizes
            
        cfg.MODEL.WEIGHTS = model_path
        predictor = DefaultPredictor(cfg)
        

        latest_file = find_latest_error_forward_file(test_directory)
        if not latest_file:
            return center_list, mask_sizes
            

        if not validate_image(latest_file):
            return center_list, mask_sizes
        

        im = cv2.imread(latest_file, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"Error: Could not read image {latest_file}")
            return center_list, mask_sizes
            
        print(f"Image shape: {im.shape}")
        

        outputs = predictor(im)
        instances = outputs["instances"]
        
        if len(instances) == 0:
            print("No objects detected")
            return center_list, mask_sizes
            
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        masks = instances.pred_masks.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        print(f"Detected {len(instances)} objects")
        

        for i, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            try:

                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if len(contours) == 0:
                    print(f"No contours found for object {i}")
                    continue
                    
                
                largest_contour = max(contours, key=cv2.contourArea)
                
                
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
        
        
        if visualize and len(instances) > 0:
            try:
                #(BGR -> RGB)
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


if __name__ == "__main__":
    test_dir = "Data/Approximate/241108_ZeroScan"
    centers, sizes = segmentation(test_dir)
    print(f"Results: {len(centers)} objects detected")
    for i, (center, size) in enumerate(zip(centers, sizes)):
        print(f"Object {i}: Center={center[:2]}, Size={size}")

import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import torch  


def get_cfg_cpu():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"  
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return cfg



def get_object_dicts(img_dir):
    import glob
    
    dataset_dicts = []
    annotations = []
    images = []

    
    json_files = glob.glob(os.path.join(img_dir, "*.json"))
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        images.extend(data.get("images", []))
        annotations.extend(data.get("annotations", []))

    
    annos_by_image_id = {}
    for anno in annotations:
        image_id = anno["image_id"]
        if image_id not in annos_by_image_id:
            annos_by_image_id[image_id] = []
        annos_by_image_id[image_id].append(anno)

    for img in images:
        record = {
            "file_name": os.path.join(img_dir, img["file_name"]),
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
        }

        objs = []
        for anno in annos_by_image_id.get(img["id"], []):
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": anno["segmentation"],
                "category_id": anno["category_id"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

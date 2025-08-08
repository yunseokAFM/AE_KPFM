import os
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import warnings
warnings.filterwarnings(action='ignore')

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


def get_cfg_cpu():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "cpu" 
    return cfg

for d in ["train", "val"]:
    if f"object_{d}" in DatasetCatalog.list():
        DatasetCatalog.remove(f"object_{d}")
    DatasetCatalog.register(
        f"object_{d}", 
        lambda d=d: get_object_dicts(os.path.join("object", d))
    )
    MetadataCatalog.get(f"object_{d}").set(thing_classes=["object"])


cfg = get_cfg_cpu()
cfg.DATASETS.TRAIN = ("object_train",)   
cfg.DATASETS.TEST = ("object_val",)      
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.OUTPUT_DIR = "output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

trainer.train()

print(f"Training completed. The final model is stored at: {os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')}")

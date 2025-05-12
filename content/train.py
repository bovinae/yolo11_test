# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

os.chdir(r"E:\work\github\bovinae\yolo11_test")
model_n = YOLO('yolo11n.pt')

if __name__ == '__main__':
    model_n.train(data='./brain-tumor.yaml',
                epochs=26,
                batch=16,
                imgsz=256,
                device=0,
                workers=0,
                scale=0.8,
                mosaic=0.8,
                mixup=0.12,
                copy_paste=0.3
    )

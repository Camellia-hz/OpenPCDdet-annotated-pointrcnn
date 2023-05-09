import cv2
import numpy as np
from pathlib import Path


idx = "000000"
path = "/home/nathan/OpenPCDet/data/kitti/training"
path = Path(path)
label_path = path / 'label_2' / f'{idx}.txt'
img_path = path / 'limage_2' / f'{idx}.png'
print(img_path)

with open(label_path, "r") as f:
    label = f.readlines()
    print(label)


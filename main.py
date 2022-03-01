from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import numpy as np
import os.path as path
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from detectron2.data import MetadataCatalog
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from progress.bar import Bar
from common.generators import ObscureGenerator


def main():
    dataset = ObscureGenerator(
        "data",
        total=3,
        seed=42,
        debug_code=1
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    metadata = MetadataCatalog.get(dataset.cfg.DATASETS.TEST[0])

    for i in range(len(dataset)):
        gt_keypoints, obscured_keypoints, ratio, debug = dataset[i]
        boxes = debug["boxes"]
        sample_image = debug["sample_image"]
        obscured_viz = debug["obscured_viz"]

        sample_keypoints = obscured_keypoints
        for j in range(len(sample_keypoints)):
            v1 = Visualizer(sample_image, metadata=metadata, scale=0.5)
            v1.draw_and_connect_keypoints(torch.from_numpy(gt_keypoints[j]))
            v1.draw_box(boxes[j])
            v1_np = v1.output.get_image()

            v2 = Visualizer(obscured_viz[j], metadata=metadata, scale=0.5)
            v2.draw_and_connect_keypoints(torch.from_numpy(obscured_keypoints[j]))
            v2.draw_text("RATIO: " + str(ratio), (np.shape(obscured_viz[j])[0] // 2, 100))
            v2.draw_box(boxes[j])
            v2_np = v2.output.get_image()

            cv2.imshow("ORIG", v1_np)
            cv2.imshow("RATIO: " + str(ratio), v2_np)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
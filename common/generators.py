# Some basic setup:
# Setup detectron2 logger
import copy

import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
setup_logger()

# import some common libraries like numpy, json, cv2 etc.
import os, json, cv2, random
import sys
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import math
ROOT_DIR = str(Path(__file__).resolve().parents[1])
print(ROOT_DIR)
sys.path.append(ROOT_DIR)

skeleton_config = {
    'config_file': 'common/keypoint_config/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
    'opts': ['MODEL.WEIGHTS', 'https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl'],
    'conf_thres': 0.8,
}

class ObscureGenerator(Dataset):
    def __init__(self, img_dir, total=20, seed=42, rand_lbound=0, rand_ubound=0.7, debug_code=0):
        assert img_dir is not None
        self.debug_code = debug_code
        self.img_dir = img_dir
        self.total = int(total)
        self.seed = seed
        self.rand_lbound = rand_lbound
        self.rand_ubound = rand_ubound

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        #torch.use_deterministic_algorithms(True)

        self.img_list = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))

        self.cfg = get_cfg()
        add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(skeleton_config['config_file'])
        self.cfg.merge_from_list(skeleton_config['opts'])

        # Set score_threshold for builtin models
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = skeleton_config['conf_thres']
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = skeleton_config['conf_thres']
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = skeleton_config['conf_thres']
        self.cfg.freeze()

        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        self.predictor = DefaultPredictor(self.cfg)

        self.obscure_ratios = []
        for i in range(len(self.img_list)):
            ls = []
            for j in range(self.total):
                ls.append(random.uniform(self.rand_lbound, self.rand_ubound))
            self.obscure_ratios.append(ls)

    def __getitem__(self, index):
        img_idx = int(index / self.total)
        ora_idx = index % self.total

        gt_img_np = cv2.imread(self.img_list[img_idx])
        with torch.no_grad():
            pred_obj = self.predictor(gt_img_np)
        gt_img_keypoints = pred_obj['instances'].to(torch.device('cpu')).pred_keypoints

        ratio = self.obscure_ratios[img_idx][ora_idx]

        gt_img_keypoints = gt_img_keypoints.numpy()

        obscured_keypoints = []
        boxes = []
        obscured_viz = []
        for keypoints_per_instance in gt_img_keypoints:
            gt_img_np_cpy = copy.deepcopy(gt_img_np)
            low_x = 999
            low_y = 999
            high_x = -999
            high_y = -999
            for coord in keypoints_per_instance:
                if coord[0] > high_x:
                    high_x = coord[0]
                if coord[0] < low_x:
                    low_x = coord[0]
                if coord[1] > high_y:
                    high_y = coord[1]
                if coord[1] < low_y:
                    low_y = coord[1]
            cutoff = low_y + (1 - ratio) * (high_y - low_y)
            for y_val in range(math.ceil(cutoff), np.shape(gt_img_np_cpy)[0]):
                gt_img_np_cpy[y_val] = np.zeros((np.shape(gt_img_np_cpy)[1], 3))

            if self.debug_code != 0:
                boxes.append((low_x, low_y, high_x, high_y))
                obscured_viz.append(gt_img_np_cpy)

            with torch.no_grad():
                pred_obj = self.predictor(gt_img_np_cpy)
            obscured_keypoints = (pred_obj['instances'].to(torch.device('cpu')).pred_keypoints).numpy()


        if self.debug_code == 0:
            return gt_img_keypoints, obscured_keypoints, ratio, None
        if self.debug_code == 1:
            debug = {
                "sample_image": gt_img_np,
                "boxes": boxes,
                "obscured_viz": obscured_viz
            }
            return gt_img_keypoints, obscured_keypoints, ratio, debug

    def __len__(self):
        return len(self.img_list) * self.total

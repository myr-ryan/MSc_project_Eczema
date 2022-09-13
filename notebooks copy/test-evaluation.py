import sys
import ast
sys.path.append("..")
sys.path.append("../src")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('keras')
import ad_seg_utils as seg_utils
import bb_eval_utils as eval_utils
import csv
import math
import albumentations as A


################################ Model loading (this part will be replaced with new data type soon) ################################
BIN_SEG = True
MODEL_NAME = '/mul_seg_model.h5'

CLASSES = ['background', 'skin', 'eczema']
WEIGHTS = np.array([1, 1, 1])
target_idx = 2

if BIN_SEG:
    MODEL_NAME = '/bin_seg_model.h5'
    CLASSES = ['background', 'skin']
    WEIGHTS = np.array([1, 1])
    target_idx = 1

BACKBONE = 'efficientnetb3'
LR = 0.0001
preprocess_input = sm.get_preprocessing(BACKBONE)

"""# Model Evaluation"""
PROJ_DIR = "/rds/general/user/zw120/home/EczemaNet/Eczema-Deep-Learning-Segmentation"
MODEL_DIR = os.path.join(PROJ_DIR, 'output');
OUTPUT_DIR = "/rds/general/user/zw120/home/EczemaNet/EczemaNet/data"

# define network parameters
n_classes = len(CLASSES)
# select training mode
activation = 'sigmoid' if n_classes == 1 else 'softmax'
#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
# define optomizer
optim = keras.optimizers.Adam(LR)
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# Set class weights for diss_loss (background: 1, skin: 1, eczema: 1)
dice_loss = sm.losses.DiceLoss(class_weights=WEIGHTS)
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# load trained segmentation model
model.load_weights(MODEL_DIR + MODEL_NAME)


# %%time
folder_dir = '/rds/general/user/zw120/home/EczemaNet/Eczema-Deep-Learning-Segmentation/data/perturbed_test_sets/adversarial_test_set_jun'
fns = os.listdir(folder_dir)
for file in fns:
    print(os.path.join(folder_dir, file))
    seg_utils.run_sigle_pred(model, os.path.join(folder_dir, file), "/rds/general/user/zw120/home/EczemaNet/EczemaNet/data/test", file, target_idx=target_idx, resize_ratio=1, preprocessing=seg_utils.get_preprocessing(preprocess_input), refno="",visno="")

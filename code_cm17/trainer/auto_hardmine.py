import sys
import os
import argparse
import logging
import json
import time
import numpy as np
import openslide
import PIL
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import DataLoader
import math
import json
import logging
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from skimage.transform import resize, rescale
import gzip
import time
import pandas as pd
import csv
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *
from data.automine_data_loader import WSIStridedPatchDataset
from models.seg_models import *
np.random.seed(0)


# python3 auto_hardmine.py /media/mak/mirlproject1/CAMELYON17/training/dataset/center_0/patient_004_node_4.tif  /media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/CM16/unet_densenet121_imagenet_pretrained_L0_20190712-173828/Model_Stage2.h5 ../configs/DenseNet121_UNET_NCRF_CM16_COORDS_CDL_AUTOMINE.json ./patient_004_node_4_mask.npy --label_path='/media/mak/mirlproject1/CAMELYON17/training/groundtruth/lesion_annotations/Mask/patient_004_node_4_mask.tif' --mask_path='/media/mak/mirlproject1/CAMELYON17/training/groundtruth/lesion_annotations/Mask/patient_004_node_4_mask.tif'

parser = argparse.ArgumentParser(description='Hardmine points from CM17 training WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('model_path', default=None, metavar='MODEL_PATH', type=str,
                    help='Path to the saved model weights file of a Keras model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('out_csv_path', default=None, metavar='OUT_CSV_PATH',
                    type=str, help='Path to the output csv file')
parser.add_argument('--mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--label_path', default=None, metavar='LABEL_PATH', type=str,
                    help='Path to the Ground-Truth label image')
parser.add_argument('--GPU', default='0,1', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=8, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--level', default=5, type=int, help='heatmap generation level,'
                    ' default 5')
parser.add_argument('--sampling_stride', default=16, type=int, help='Sampling pixels in tissue mask,'
                    ' default 16')
parser.add_argument('--roi_masking', default=True, type=int, help='Sample pixels from tissue mask region,'
                    ' default True, points are not sampled from glass region')


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum



def get_probs_map(model, dataloader):
    """
    Generate probability map
    """
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    count_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)
    batch_size = dataloader.batch_size
    map_x_size = dataloader.dataset._mask.shape[0]
    map_y_size = dataloader.dataset._mask.shape[1]
    level = dataloader.dataset._level
    factor = dataloader.dataset._sampling_stride
    flip = dataloader.dataset._flip
    rotate = dataloader.dataset._rotate
    wsi_name = os.path.basename(dataloader.dataset._wsi_path)   
    down_scale = 1.0 / pow(2, level)

    count = 0
    time_now = time.time()
    probs_map = []

    DICE_THRESHOLD = 0.90
    # label_mask is not utilized     
    print ('Started Mining')
    try:
        model = multi_gpu_model(model, gpus=2, cpu_merge=False)
        print("Inference on multiple GPUs..")
    except:
        print("Inference on single GPU or CPU..")

    for (image_patches, x_coords, y_coords, label_patches) in dataloader:

        image_patches = image_patches.cpu().data.numpy()
        label_patches = label_patches.cpu().data.numpy()
        x_coords = x_coords.cpu().data.numpy()*pow(2,level)
        y_coords = y_coords.cpu().data.numpy()*pow(2,level)

        y_preds = model.predict(image_patches, batch_size=batch_size, verbose=0, steps=None)

        for i in range(batch_size):
            y_pred_mask = labelthreshold(y_preds[i][:,:,1],  threshold=0.45)
            y_true_mask = label_patches[i]
            dice_score = dice(y_pred_mask, y_true_mask)
            if dice_score < DICE_THRESHOLD:
                # print (wsi_name, str(x_coords[i]), str(y_coords[i]), str(dice_score))
                probs_map.append((wsi_name, str(x_coords[i]), str(y_coords[i]), str(dice_score)))
                # imshow(normalize_minmax(image_patches[i]), y_pred_mask, y_true_mask)
        count += 1
        time_spent = time.time() - time_now
        time_now = time.time()
        print ('{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))

        # logging.info(
        #     '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
        #     .format(
        #         time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
        #         dataloader.dataset._rotate, count, num_batch, time_spent))
    # imshow(count_map)
    return probs_map

def make_dataloader(args, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size']
    dataloader = DataLoader(WSIStridedPatchDataset(args.wsi_path, args.mask_path,
                            args.label_path,
                            image_size=cfg['image_size'],
                            normalize=True, flip=flip, rotate=rotate,
                            level=args.level, sampling_stride=args.sampling_stride, roi_masking=args.roi_masking),
                            batch_size=batch_size, num_workers=args.num_workers, drop_last=True)
    return dataloader

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    core_config =  tf.compat.v1.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    session = tf.compat.v1.Session(config=core_config) 
    tf.compat.v1.keras.backend.set_session(session)

    # Instantiate the base model (or "template" model).
    # We recommend doing this with under a CPU device scope,
    # so that the model's weights are hosted on CPU memory.
    # Otherwise they may end up hosted on a GPU, which would
    # complicate weight sharing.
    # with tf.device('/cpu:0'):
    model = unet_densenet121((None, None), weights=None)
    
    model.load_weights(args.model_path)
    print ("Loaded Model Weights")

    save_dir = os.path.dirname(args.out_csv_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataloader = make_dataloader(
        args, cfg, flip='NONE', rotate='NONE')
    probs_map = get_probs_map(model, dataloader)


    with open(args.out_csv_path, 'w') as out:
        csv_out = csv.writer(out)
        for row in probs_map:
            csv_out.writerow(row)        

def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

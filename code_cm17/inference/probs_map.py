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
from skimage.transform import resize, rescale
import gzip
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *
from data.data_loader import WSIStridedPatchDataset
from models.seg_models import *
np.random.seed(0)


# python3 probs_map.py /media/mak/mirlproject1/CAMELYON17/training/dataset/center_0/patient_004_node_4.tif  /media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/CM16/unet_densenet121_imagenet_pretrained_L0_20190712-173828/Model_Stage2.h5 ./configs/UNET_FCN.json ../../../predictions/DenseNet-121_UNET/patient_004_node_4_mask.npy

parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('model_path', default=None, metavar='MODEL_PATH', type=str,
                    help='Path to the saved model weights file of a Keras model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--label_path', default=None, metavar='LABEL_PATH', type=str,
                    help='Path to the Ground-Truth label image')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=5, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=1, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')
parser.add_argument('--level', default=5, type=int, help='heatmap generation level,'
                    ' default 5')
parser.add_argument('--sampling_stride', default=32, type=int, help='Sampling pixels in tissue mask,'
                    ' default 32')
parser.add_argument('--roi_masking', default=True, type=int, help='Sample pixels from tissue mask region,'
                    ' default True, points are not sampled from glass region')


def transform_prob(data, flip, rotate):
    """
    Do inverse data augmentation
    """
    if flip == 'FLIP_LEFT_RIGHT':
        data = np.fliplr(data)

    if rotate == 'ROTATE_90':
        data = np.rot90(data, 3)

    if rotate == 'ROTATE_180':
        data = np.rot90(data, 2)

    if rotate == 'ROTATE_270':
        data = np.rot90(data, 1)

    return data

def get_index(coord_ax, probs_map_shape_ax, grid_ax):
    """
    This function checks whether coordinates are within the WSI
    """
    # print (coord_ax, probs_map_shape_ax, grid_ax)
    _min = grid_ax//2
    _max = grid_ax//2

    ax_min = coord_ax - _min
    while ax_min < 0:
        _min -= 1
        ax_min += 1

    ax_max = coord_ax + _max
    while ax_max > probs_map_shape_ax:
        _max -= 1
        ax_max -= 1

    return _min, _max


def get_probs_map(model, dataloader):
    """
    Generate probability map
    """
    eps = 0.0001
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
    down_scale = 1.0 / pow(2, level)

    count = 0
    time_now = time.time()
    # label_mask is not utilized     
    for (image_patches, x_coords, y_coords, label_patches) in dataloader:

        image_patches = image_patches.cpu().data.numpy()
        label_patches = label_patches.cpu().data.numpy()
        x_coords = x_coords.cpu().data.numpy()
        y_coords = y_coords.cpu().data.numpy()

        # start = time.time()
        y_preds = model.predict(image_patches, batch_size=batch_size, verbose=1, steps=None)
        # end = time.time()
        # print('Elapsed Inference Time', (end - start))

        # print (image_patches[0].shape, y_preds[0].shape)  
        # imshow (normalize_minmax(image_patches[0]),label_patches[0], y_preds[0][:,:,0], y_preds[0][:,:,1], np.argmax(y_preds[0], axis=2))

        for i in range(batch_size):
            # start = time.time()
            y_preds_transformed = transform_prob(y_preds[i], flip, rotate)
            # img_patch_rescaled = rescale(image_patches[i], down_scale, anti_aliasing=True)
            y_preds_rescaled = rescale(y_preds_transformed, down_scale, anti_aliasing=False)
            # imshow(normalize_minmax(image_patches[i]), label_patches[i], y_preds[i][:,:,1], np.argmax(y_preds[i], axis=2), title=['Image', 'Ground-Truth', 'Heat-Map', 'Predicted-Label-Map'])            
            # imshow(normalize_minmax(img_patch_rescaled), y_preds_rescaled[:,:,1], np.argmax(y_preds_rescaled, axis=2), title=['Rescaled-Image', 'Rescaled-Predicted-Heat-Map', 'Rescaled-Predicted-Label-Map'])
            xmin, xmax = get_index(x_coords[i], map_x_size, factor)
            ymin, ymax = get_index(y_coords[i], map_y_size, factor)
            # print (xmin, xmax, ymin, ymax)   
            probs_map[x_coords[i] - xmin: x_coords[i] + xmax, y_coords[i] - ymin: y_coords[i] + ymax] =\
            y_preds_rescaled[0:xmin+xmax, 0:ymin+ymax, 1]
            count_map[x_coords[i] - xmin: x_coords[i] + xmax, y_coords[i] - ymin: y_coords[i] + ymax] +=\
            np.ones_like(y_preds_rescaled[0:xmin+xmax, 0:ymin+ymax, 1])
            # end = time.time()
            # print('Elapsed post inference time', (end - start))
    
        count += 1
        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))
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

    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    session =tf.Session(config=core_config) 
    K.set_session(session)

    model = unet_densenet121((None, None), weights=None)
    model.load_weights(args.model_path)
    print ("Loaded Model Weights")

    save_dir = os.path.dirname(args.probs_map_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not args.eight_avg:
        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(model, dataloader)
    else:        
        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
        probs_map = np.zeros(dataloader.dataset._mask.shape)

        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        probs_map /= 8

    np.save(args.probs_map_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

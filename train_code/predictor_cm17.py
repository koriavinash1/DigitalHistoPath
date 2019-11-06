import os
import numpy as np
import openslide
import PIL
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import json
import logging
import time
from skimage.transform import resize
from torch.nn import DataParallel
from utils import *
from models import *
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

def BinMorphoProcessMask(mask):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    return image_open 

def TissueMaskGeneration(slide_obj, level, RGB_min=50):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])
    img_HSV = rgb2hsv(img_RGB)
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return tissue_mask

class GridWSIStridedPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, wsi_path, mask_path, label_mask_path=None, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE',                
                 level=6, stride_factor=4, roi_masking=True):
        """
        Initialize the data producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format OR None
            label_mask_path: string, path to ground-truth label mask path in tif file or
                            None (incase of Normal WSI or test-time)
            image_size: int, size of the image before splitting into grid, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
            level: Level to extract the WSI tissue mask
            roi_masking: True: Multiplies the strided WSI with tissue mask to eliminate white spaces,
                                False: Ensures inference is done on the entire WSI   
            stride_factor: Number of pixels to skip in the tissue mask, basically it's the overlap
                            fraction when patches are extracted from WSI during inference.
                            stride=1 -> consecutive pixels are utilized
                            stride= image_size/pow(2, level) -> non-overalaping patches 
        """
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._label_mask_path = label_mask_path
        self._image_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._level = level
        self._stride_factor = stride_factor
        self._roi_masking = roi_masking
        self._preprocess()

    def _preprocess(self):
        self._slide = openslide.OpenSlide(self._wsi_path)
        X_slide, Y_slide = self._slide.level_dimensions[0]

        if self._label_mask_path is not None:
            self._label_slide = openslide.OpenSlide(self._label_mask_path)
        
        if self._mask_path is not None:
            mask_file_name = os.path.basename(self._mask_path)
            if mask_file_name.endswith('.npy'):
                self._mask = np.load(self._mask_path)
            if mask_file_name.endswith('.tif'):
                mask_obj = openslide.OpenSlide(self._mask_path)
                self._mask = np.array(mask_obj.read_region((0, 0),
                       level,
                       mask_obj.level_dimensions[level]).convert('L')).T
        else:
            # Generate tissue mask on the fly    
            self._mask = TissueMaskGeneration(self._slide, self._level)
           
        # morphological operations ensure the holes are filled in tissue mask
        # and minor points are aggregated to form a larger chunk         
        self._mask = BinMorphoProcessMask(np.uint8(self._mask))

        X_mask, Y_mask = self._mask.shape
        # print (self._mask.shape, np.where(self._mask>0))
        # imshow(self._mask.T)
        # cm17 dataset had issues with images being power's of 2 precisely        
        if X_slide // X_mask != Y_slide // Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self._resolution = np.round(X_slide * 1.0 / X_mask)
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                            ' {}'.format(self._resolution))
             
        # all the idces for tissue region from the tissue mask  
        self._strided_mask =  np.ones_like(self._mask)
        factor = self._stride_factor
        ones_mask = np.zeros_like(self._mask)
        ones_mask[::factor, ::factor] = self._strided_mask[::factor, ::factor]
        if self._roi_masking:
            self._strided_mask = ones_mask*self._mask    
        else:
            self._strided_mask = ones_mask    

        self._X_idcs, self._Y_idcs = np.where(self._strided_mask)        
        self._idcs_num = len(self._X_idcs)

    def __len__(self):        
        return self._idcs_num 

    def save_get_mask(self, save_path):
        np.save(save_path, self._mask)

    def get_mask(self):
        return self._mask

    def get_strided_mask(self):
        return self._strided_mask
    
    def __getitem__(self, idx):
        x_coord, y_coord = self._X_idcs[idx], self._Y_idcs[idx]
    
        x = int(x_coord * self._resolution - self._image_size//2)
        y = int(y_coord * self._resolution - self._image_size//2)

        img = self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB')
        
        if self._label_mask_path is not None:
            label_img = np.array(self._label_slide.read_region(
                (x, y), 0, (self._image_size, self._image_size)).convert('L'))
        else:
            label_img = np.zeros((self._image_size, self._image_size), dtype=np.uint8)
        
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_img = np.fliplr(label_img)
            
        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)
            label_img = np.rot(label_img, 1)
            
        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)
            label_img = np.rot(label_img, 2)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)
            label_img = np.rot(label_img, 3)

        # PIL image:   H x W x C
        img = np.array(img, dtype=np.float32)

        if self._normalize:
            img = (img - 128.0)/128.0
   
        return (img, x_coord, y_coord, label_img)

def get_index(coord_ax, probs_map_shape_ax, grid_ax):
    """
    This function checks whether coordinates are within the WSI
    """
    # print (coord_ax, probs_map_shape_ax, grid_ax)
    _min = 0
    _max = grid_ax

    # ax_min = coord_ax - _min
    # while ax_min < 0:
    #     _min -= 1
    #     ax_min += 1

    ax_max = coord_ax + _max
    while ax_max > probs_map_shape_ax:
        _max -= 1
        ax_max -= 1

    return _min, _max


def get_probs_map(model, dataloader):
    """
    Generate probability map
    """
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)
    batch_size = dataloader.batch_size
    map_x_size = dataloader.dataset._mask.shape[0]
    map_y_size = dataloader.dataset._mask.shape[1]
    factor = dataloader.dataset._stride_factor

    count = 0
    time_now = time.time()
    # label_mask is not utilized     
    for (image_patches, x_coords, y_coords, label_patches) in dataloader:
        image_patches = image_patches.cpu().data.numpy()
        label_patches = label_patches.cpu().data.numpy()
        x_coords = x_coords.cpu().data.numpy()
        y_coords = y_coords.cpu().data.numpy()
        y_preds = model.predict(image_patches, batch_size=batch_size, verbose=1, steps=None)

        print (image_patches[0].shape, y_preds[0].shape)  
        imshow (normalize_minmax(image_patches[0]),label_patches[0], y_preds[0][:,:,0], y_preds[0][:,:,1], np.argmax(y_preds[0], axis=2))          
        # for i in range(batch_size):
        #     _, xmax = get_index(x_coords[i], map_x_size, factor)
        #     _, ymax = get_index(y_coords[i], map_y_size, factor)   
        #     probs_map[x_coords[i] : x_coords[i]+xmax, y_coords[i]: y_coords[i]+ymax] =\
        #     probs_enlarged[0:xmax, 0:ymax]
        count += 1
        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))
    return probs_map

def make_dataloader(args, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size']
    
    dataloader = DataLoader(GridWSIStridedPatchDataset(args.wsi_path, args.label_mask_path,
                            args.label_mask_path,
                            image_size=cfg['image_size'],
                            normalize=True, flip=flip, rotate=rotate,
                            level=args.level, stride_factor=args.stride_factor, roi_masking=args.roi_masking),
                            batch_size=batch_size, num_workers=num_workers, drop_last=True)
    return dataloader

class arguments(object):
    def __init__(self, wsi_path, model_path, cfg_path, mask_path, label_mask_path, probs_map_path, GPU, num_workers, eight_avg,
                level, stride_factor, roi_masking):
        self.wsi_path = wsi_path
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.mask_path = mask_path
        self.label_mask_path = label_mask_path
        self.probs_map_path = probs_map_path
        self.GPU = GPU
        self.num_workers = num_workers
        self.eight_avg = eight_avg
        self.level = level
        self.stride_factor = stride_factor
        self.roi_masking = roi_masking

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    model = unet_densenet121((None, None), weights=None)
    model.load_weights(args.model_path)
    print ("Loaded Model Weights")
  
    if not eight_avg:
        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(model, dataloader)
    else:        
        probs_map = np.zeros(dataloader.dataset._mask.shape)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
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

if __name__ == '__main__':  
    # Test code for faster inference
    wsi_path = "/media/mak/mirlproject1/CAMELYON17/training/dataset/center_0/patient_004_node_4.tif"
    # label_mask_path is the groundtruth-> This is utilized for training purpose dataloader
    label_mask_path = "/media/mak/mirlproject1/CAMELYON17/training/groundtruth/lesion_annotations/Mask/patient_004_node_4_mask.tif"
    # label_mask_path = None
    mask_path = None
    model_path = '/media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/unet_densenet121_imagenet_pretrained_L0_20190711-182836/Model_Stage3.h5'
    cfg_path = './UNET_FCN.json'
    probs_map_path = './patient_004_node_4.npy'
    batch_size = 1
    num_workers = 0
    GPU = '1'
    eight_avg = 0
    level = 5
    stride_factor = 1
    roi_masking = True


    args = arguments(wsi_path, model_path, cfg_path, mask_path, label_mask_path, probs_map_path, GPU, num_workers, eight_avg,
                    level, stride_factor, roi_masking)

    run(args)
    plt.imshow(np.load(args.probs_map_path).T, cmap='jet')
    print (args.probs_map_path[:-4]+'.png')
    plt.savefig(args.probs_map_path[:-4]+'.png')

    # dataset_obj = GridWSIStridedPatchDataset(wsi_path, mask_path,
    #                                         label_mask_path,
    #                                         image_size=256,
    #                                         normalize=True,
    #                                         flip=None, rotate=None,
    #                                         level=6, stride_factor=4, roi_masking=True)

    # print (dataloader.dataset.__len__(), dataloader.__len__())
    # i = 0
    # start_time = time.time()
    # # # imshow(dataset_obj.get_mask(), dataset_obj.get_strided_mask())
    # for (data, x_mask, y_mask, label) in dataloader:
    #     image_patches = data.cpu().data.numpy()
    #     label_patches = label.cpu().data.numpy()
    #     x_coords = x_mask.cpu().data.numpy()
    #     y_coords = y_mask.cpu().data.numpy()
    #     # print (image_patches.shape, image_patches.dtype, label_patches.shape, label_patches.dtype)
    #     # print (x_coords, y_coords)
    #     # For display 
    #     input_map = normalize_minmax(image_patches[0])
    #     label_map = label_patches[0]
    #     if np.sum(label_map) > 0:
    #         print (np.sum(label_map)/np.prod(label_map.shape))
    #         imshow(input_map, label_map)
    #         i += 1
    #         if i == 10:
    #             elapsed_time = time.time() - start_time
    #             print ("Elapsed Time", np.round(elapsed_time, decimals=2), "seconds")
    #             break

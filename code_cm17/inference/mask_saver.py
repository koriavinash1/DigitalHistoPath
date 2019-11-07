from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import glob
import random
import imgaug
from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import openslide
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, concatenate, Concatenate, UpSampling2D, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # noqa

import sklearn.metrics
import io
import itertools
from six.moves import range

import time
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from models.seg_models import get_inception_resnet_v2_unet_softmax, unet_densenet121
from models.deeplabv3p_original import Deeplabv3
# Random Seeds
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)
import gc
import pandas as pd

import tifffile 
import skimage.io as io


# In[50]:


# Image Helper Functions
def imsave(*args, **kwargs):
     """
     Concatenate the images given in args and saves them as a single image in the specified output destination.
     Images should be numpy arrays and have same dimensions along the 0 axis.
     imsave(im1,im2,out="sample.png")
     """
     args_list = list(args)
     for i in range(len(args_list)):
         if type(args_list[i]) != np.ndarray:
             print("Not a numpy array")
             return 0
         if len(args_list[i].shape) == 2:
             args_list[i] = np.dstack([args_list[i]]*3)
             if args_list[i].max() == 1:
                args_list[i] = args_list[i]*255

     out_destination = kwargs.get("out",'')
     try:
         concatenated_arr = np.concatenate(args_list,axis=1)
         im = Image.fromarray(np.uint8(concatenated_arr))
     except Exception as e:
         print(e)
         import ipdb; ipdb.set_trace()
         return 0
     if out_destination:
         print(f"Saving to {out_destination}")
         im.save(out_destination)
     else:
        return im

def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_off = kwargs.get('axis_off','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
            if axis_off: 
              plt.axis('off')  
    plt.show()
def normalize_minmax(data):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max-_min)!=0:
        img = (data - _min) / (_max-_min)
    else:
        img = np.zeros_like(data)            
    return img

# Functions
def BinMorphoProcessMask(mask,level):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    if level == 2:
        kernel = np.ones((60, 60), dtype=np.uint8)
    elif level == 3:
        kernel = np.ones((35, 35), dtype=np.uint8)
    else:
        raise ValueError
    image = cv2.dilate(image_open,kernel,iterations = 1)
    return image

def get_bbox(cont_img, rgb_image=None):
    temp_img = np.uint8(cont_img.copy())
    _,contours, _ = cv2.findContours(temp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (0, 0, 255)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    for x, y, h, w in bounding_boxes:
        rgb_contour = cv2.rectangle(rgb_contour,(x,y),(x+h,y+w),(0,255,0),2)
    return bounding_boxes, rgb_contour

def get_all_bbox_masks(mask, stride_factor):
    """
    Find the bbox and corresponding masks
    """
    bbox_mask = np.zeros_like(mask)
    bounding_boxes, _ = get_bbox(mask)
    y_size, x_size = bbox_mask.shape
    for x, y, h, w in bounding_boxes:
        x_min = x - stride_factor
        x_max = x + h + stride_factor
        y_min = y - stride_factor
        y_max = y + w + stride_factor
        if x_min < 0: 
         x_min = 0
        if y_min < 0: 
         y_min = 0
        if x_max > x_size: 
         x_max = x_size - 1
        if y_max > y_size: 
         y_max = y_size - 1      
        bbox_mask[y_min:y_max, x_min:x_max]=1
    return bbox_mask

def get_all_bbox_masks_with_stride(mask, stride_factor):
    """
    Find the bbox and corresponding masks
    """
    bbox_mask = np.zeros_like(mask)
    bounding_boxes, _ = get_bbox(mask)
    y_size, x_size = bbox_mask.shape
    for x, y, h, w in bounding_boxes:
        x_min = x - stride_factor
        x_max = x + h + stride_factor
        y_min = y - stride_factor
        y_max = y + w + stride_factor
        if x_min < 0: 
         x_min = 0
        if y_min < 0: 
         y_min = 0
        if x_max > x_size: 
         x_max = x_size - 1
        if y_max > y_size: 
         y_max = y_size - 1      
        bbox_mask[y_min:y_max:stride_factor, x_min:x_max:stride_factor]=1
        
    return bbox_mask

def find_largest_bbox(mask, stride_factor):
    """
    Find the largest bounding box encompassing all the blobs
    """
    y_size, x_size = mask.shape
    x, y = np.where(mask==1)
    bbox_mask = np.zeros_like(mask)
    x_min = np.min(x) - stride_factor
    x_max = np.max(x) + stride_factor
    y_min = np.min(y) - stride_factor
    y_max = np.max(y) + stride_factor
    
    if x_min < 0: 
     x_min = 0
    
    if y_min < 0: 
     y_min = 0

    if x_max > x_size: 
     x_max = x_size - 1
    
    if y_min > y_size: 
     y_max = y_size - 1    
    
    bbox_mask[x_min:x_max, y_min:y_max]=1
    return bbox_mask
    
    
def TissueMaskGeneration(slide_obj, level, RGB_min=50):
    img_RGB = slide_obj.read_region((0, 0),level,slide_obj.level_dimensions[level])
    img_RGB = np.transpose(np.array(img_RGB.convert('RGB')),axes=[1,0,2])
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
    # r = img_RGB[:,:,0] < 235
    # g = img_RGB[:,:,1] < 210
    # b = img_RGB[:,:,2] < 235
    # tissue_mask = np.logical_or(r,np.logical_or(g,b))
    return tissue_mask 
def TissueMaskGenerationPatch(patchRGB):
    '''
    Returns mask of tissue that obeys the threshold set by paip
    '''
    r = patchRGB[:,:,0] < 235
    g = patchRGB[:,:,1] < 210
    b = patchRGB[:,:,2] < 235
    tissue_mask = np.logical_or(r,np.logical_or(g,b))
    return tissue_mask 
    
def TissueMaskGeneration_BIN(slide_obj, level):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])    
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    img_S = img_HSV[:, :, 1]
    _,tissue_mask = cv2.threshold(img_S, 0, 255, cv2.THRESH_BINARY)
    return np.array(tissue_mask)

def TissueMaskGeneration_BIN_OTSU(slide_obj, level):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])    
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    img_S = img_HSV[:, :, 1]
    _,tissue_mask = cv2.threshold(img_S, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return np.array(tissue_mask)

def labelthreshold(image, threshold=0.5):
    np.place(image,image>=threshold, 1)
    np.place(image,image<threshold, 0)
    return np.uint8(image)

def calc_jacc_score(x,y,smoothing=1):
    for var in [x,y]:
        np.place(var,var==255,1)
    
    numerator = np.sum(x*y)
    denominator = np.sum(np.logical_or(x,y))
    return (numerator+smoothing)/(denominator+smoothing)



# In[41]:


# DataLoader Implementation
class WSIStridedPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, wsi_path, mask_path, label_path=None, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE',                
                 level=5, sampling_stride=16, roi_masking=True):
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
            sampling_stride: Number of pixels to skip in the tissue mask, basically it's the overlap
                            fraction when patches are extracted from WSI during inference.
                            stride=1 -> consecutive pixels are utilized
                            stride= image_size/pow(2, level) -> non-overalaping patches 
        """
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._label_path = label_path
        self._image_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._level = level
        self._sampling_stride = sampling_stride
        self._roi_masking = roi_masking
        
        self._preprocess()

    def _preprocess(self):
        self._slide = openslide.OpenSlide(self._wsi_path)
        
        if self._label_path is not None:
            self._label_slide = openslide.OpenSlide(self._label_path)
        
        X_slide, Y_slide = self._slide.level_dimensions[0]
        print("Image dimensions: (%d,%d)" %(X_slide,Y_slide))
        
        factor = self._sampling_stride

        
        if self._mask_path is not None:
            mask_file_name = os.path.basename(self._mask_path)
            if mask_file_name.endswith('.tiff'):
                mask_obj = openslide.OpenSlide(self._mask_path)
                self._mask = np.array(mask_obj.read_region((0, 0),
                       self._level,
                       mask_obj.level_dimensions[self._level]).convert('L')).T
                np.place(self._mask,self._mask>0,255)
        else:
            # Generate tissue mask on the fly    
            
            self._mask = TissueMaskGeneration(self._slide, self._level)
        # morphological operations ensure the holes are filled in tissue mask
        # and minor points are aggregated to form a larger chunk         

        self._mask = BinMorphoProcessMask(np.uint8(self._mask),self._level)
        # self._all_bbox_mask = get_all_bbox_masks(self._mask, factor)
        # self._largest_bbox_mask = find_largest_bbox(self._mask, factor)
        # self._all_strided_bbox_mask = get_all_bbox_masks_with_stride(self._mask, factor)

        X_mask, Y_mask = self._mask.shape
        # print (self._mask.shape, np.where(self._mask>0))
        # imshow(self._mask.T)
        # cm17 dataset had issues with images being power's of 2 precisely        
#         if X_slide != X_mask or Y_slide != Y_mask:
        print('Mask (%d,%d) and Slide(%d,%d) '%(X_mask,Y_mask,X_slide,Y_slide))
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
        ones_mask = np.zeros_like(self._mask)
        ones_mask[::factor, ::factor] = self._strided_mask[::factor, ::factor]
        
        
        if self._roi_masking:
            self._strided_mask = ones_mask*self._mask   
            # self._strided_mask = ones_mask*self._largest_bbox_mask   
            # self._strided_mask = ones_mask*self._all_bbox_mask 
            # self._strided_mask = self._all_strided_bbox_mask  
        else:
            self._strided_mask = ones_mask  
        # print (np.count_nonzero(self._strided_mask), np.count_nonzero(self._mask[::factor, ::factor]))
        # imshow(self._strided_mask.T, self._mask[::factor, ::factor].T)
        # imshow(self._mask.T, self._strided_mask.T)
 
        self._X_idcs, self._Y_idcs = np.where(self._strided_mask)        
        self._idcs_num = len(self._X_idcs)

    def __len__(self):        
        return self._idcs_num 

    def save_scaled_imgs(self):
        scld_dms = self._slide.level_dimensions[self._level]
        self._slide_scaled = self._slide.read_region((0,0),self._level,scld_dms)
        
        if self._label_path is not None:
            self._label_scaled = np.array(self._label_slide.read_region((0,0),4,scld_dms).convert('L'))
            np.place(self._label_scaled,self._label_scaled>0,255)
        
    def save_get_mask(self, save_path):
        np.save(save_path, self._mask)

    def get_mask(self):
        return self._mask

    def get_strided_mask(self):
        return self._strided_mask
    
    def __getitem__(self, idx):
        x_coord, y_coord = self._X_idcs[idx], self._Y_idcs[idx]
        
        x_max_dim,y_max_dim = self._slide.level_dimensions[0]

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)    

        x = int(x_coord * self._resolution - self._image_size//2)
        y = int(y_coord * self._resolution - self._image_size//2)    
#         x = int(x_coord * self._resolution)
#         y = int(y_coord * self._resolution)    
        
        #If Image goes out of bounds
        if x>(x_max_dim - image_size):
            x = x_max_dim - image_size
        elif x<0:
            x = 0
        if y>(y_max_dim - image_size):
            y = y_max_dim - image_size
        elif y<0:
            y = 0
    
        #Converting pil image to np array transposes the w and h
        img = np.transpose(self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB'),[1,0,2])
        
        if self._label_path is not None:
            label_img = self._label_slide.read_region(
                (x, y), 0, (self._image_size, self._image_size)).convert('L')
        else:
            #print('No label img')
            label_img = Image.fromarray(np.zeros((self._image_size, self._image_size), dtype=np.uint8))
        
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
            
        if self._rotate == 'ROTATE_90':
            img = img.transpose(Image.ROTATE_90)
            label_img = label_img.transpose(Image.ROTATE_90)
            
        if self._rotate == 'ROTATE_180':
            img = img.transpose(Image.ROTATE_180)
            label_img = label_img.transpose(Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(Image.ROTATE_270)
            label_img = label_img.transpose(Image.ROTATE_270)

        # PIL image:   H x W x C
        img = np.array(img, dtype=np.float32)
        label_img = np.array(label_img, dtype=np.uint8)
        np.place(label_img, label_img>0, 255)

        if self._normalize:
            img = (img - 128.0)/128.0
   
        return (img, x, y, label_img)
image_size = 1024
sampling_stride = 512
batch_size = 1
wsi_paths = glob.glob('../../data/raw-data/test/*svs')
wsi_paths.sort()
wsi_paths = wsi_paths[7:]
print("Total %d" %len(wsi_paths))
for i,wsi_path in enumerate(wsi_paths):
    print(i,'/', len(wsi_paths),wsi_path)
    label_path=None
    wsi_obj = openslide.OpenSlide(wsi_path)
    x_max_dim,y_max_dim = wsi_obj.level_dimensions[0]
    scld_dms = wsi_obj.level_dimensions[2]

    if len(wsi_obj.level_dimensions) == 3:
        level = 2
    elif len(wsi_obj.level_dimensions) == 4:
        level = 3
    
    scale = lambda x: cv2.resize(x,tuple(reversed(scld_dms))).T
    mask_path = None
    start_time = time.time()
    dataset_obj = WSIStridedPatchDataset(wsi_path, 
                                        mask_path,
                                        label_path,
                                        image_size=image_size,
                                        normalize=True,
                                        flip=None, rotate=None,
                                        level=level, sampling_stride=sampling_stride//16, roi_masking=True)

    dataloader = DataLoader(dataset_obj, batch_size=batch_size, num_workers=0, drop_last=True)
    dataset_obj.save_scaled_imgs()
    out_file = wsi_path.split('/')[-1].split('.')[0]
    # out_file = sample_id

    print(dataset_obj.get_mask().shape)
    st_im = dataset_obj.get_strided_mask()
    mask_im = np.dstack([dataset_obj.get_mask().T]*3).astype('uint8')*255
    st_im = np.dstack([dataset_obj.get_strided_mask().T]*3).astype('uint8')*255
    im_im = np.array(dataset_obj._slide_scaled.convert('RGB'))
    ov_im = st_im/2 + im_im/2
    ov_im_mask= mask_im/2 + im_im/2
    out_file = os.path.join('../../results/saved_imgs/test_mask_ref','ref_'+out_file)+'.png'
    imsave(ov_im.astype('uint8'),ov_im_mask.astype('uint8'),(im_im),out=out_file)

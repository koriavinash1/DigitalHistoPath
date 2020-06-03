from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import glob
import random
import time
import imgaug
from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import numpy as np 
from six.moves import range
import openslide
import tensorflow as tf
from torchvision import transforms  # noqa
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *

# DataLoader Implementation
class AutomineDataGeneratorCoordFly(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, wsi_path, mask_path, coord_path, image_size=(768, 768), batch_size=32, n_classes=2, n_channels=3,
                  shuffle=True, level=0, transform=None):
        'Initialization'
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.coord_path = coord_path
        self.image_size = image_size
        self.n_channels = n_channels
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self.transform = transform
        self.shuffle = shuffle
        self.level = level
        self.coords = []
        f = open(os.path.join(self.coord_path))
        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            x_center, y_center = int(x_center), int(y_center)
            self.coords.append((pid, x_center, y_center))
        f.close()
        self._num_image = len(self.coords)            
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self._num_image / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            random.shuffle(self.coords)

    def _get_one_hot(self, targets):
        res = np.eye(self.n_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[self.n_classes])

    def _normalize_image(self, image):
        # Normalize Image
        image = (image - 128.0)/128.0
        return image
    
    def _augmentation(self, image, mask):
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = self.transform.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        image = np.array(self._color_jitter(Image.fromarray(image)))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        # label = label.astype(np.bool)
        return image, mask
  
    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y = np.empty((self.batch_size, *self.image_size, self.n_classes))

        for i in range(self.batch_size):
            pid, x_center, y_center = self.coords[(index)*self.batch_size+i]        
            # Generate data
            x_top_left = int(int(x_center) - self.image_size[0] / 2)
            y_top_left = int(int(y_center) - self.image_size[1] / 2)
            image_opslide = openslide.OpenSlide(os.path.join(self.wsi_path, pid + '.tif'))
            image = image_opslide.read_region(
                (x_top_left, y_top_left), self.level,
                (self.image_size[0], self.image_size[1])).convert('RGB')
            if pid.split('_')[0] == 'Tumor':
                mask_opslide = openslide.OpenSlide(os.path.join(self.mask_path, pid.lower() + '.tif'))
                mask = mask_opslide.read_region(
                    (x_top_left, y_top_left), self.level,
                    (self.image_size[0], self.image_size[1])).convert('L')
            else:
                mask = np.zeros((self.image_size[0], self.image_size[1]))
            image = np.asarray(image)
            mask =  np.asarray(mask,dtype=np.uint8)
            mask = self._get_one_hot(mask)
            if self.transform:
                image, mask = self._augmentation(image, mask)
            image = self._normalize_image(image)
            X[i,] = image
            y[i,] = mask
        return X, y


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
        X_slide, Y_slide = self._slide.level_dimensions[0]
        factor = self._sampling_stride

        if self._label_path is not None:
            print (self._label_path)
            self._label_slide = openslide.OpenSlide(self._label_path)
        
        if self._mask_path is not None:
            mask_file_name = os.path.basename(self._mask_path)
            if mask_file_name.endswith('.npy'):
                self._mask = np.load(self._mask_path)
            if mask_file_name.endswith('.tif'):
                mask_obj = openslide.OpenSlide(self._mask_path)
                self._mask = np.array(mask_obj.read_region((0, 0),
                       self._level,
                       mask_obj.level_dimensions[self._level]).convert('L')).T
        else:
            # Generate tissue mask on the fly    
            self._mask = TissueMaskGeneration(self._slide, self._level)
           
        # morphological operations ensure the holes are filled in tissue mask
        # and minor points are aggregated to form a larger chunk         

        self._mask = BinMorphoProcessMask(np.uint8(self._mask))
        # self._all_bbox_mask = get_all_bbox_masks(self._mask, factor)
        # self._largest_bbox_mask = find_largest_bbox(self._mask, factor)
        self._all_strided_bbox_mask = get_all_bbox_masks_with_stride(self._mask, factor)

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
        ones_mask = np.zeros_like(self._mask)
        ones_mask[::factor, ::factor] = self._strided_mask[::factor, ::factor]
        if self._roi_masking:
            # self._strided_mask = ones_mask*self._mask   
            # self._strided_mask = ones_mask*self._largest_bbox_mask   
            # self._strided_mask = ones_mask*self._all_bbox_mask 
            self._strided_mask = self._all_strided_bbox_mask  
        else:
            self._strided_mask = ones_mask  
        # print (np.count_nonzero(self._strided_mask), np.count_nonzero(self._mask[::factor, ::factor]))
        # imshow(self._strided_mask.T, self._mask[::factor, ::factor].T)
        # imshow(self._mask.T, self._strided_mask.T)
 
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

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)    

        x = int(x_coord * self._resolution - self._image_size//2)
        y = int(y_coord * self._resolution - self._image_size//2)    

        img = self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB')
        
        if self._label_path is not None:
            label_img = self._label_slide.read_region(
                (x, y), 0, (self._image_size, self._image_size)).convert('L')
        else:
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

        if self._normalize:
            img = (img - 128.0)/128.0
   
        return (img, x_coord, y_coord, label_img)


if __name__ == '__main__':
    # Training Data Configuration
    train_coord_path = '/media/mak/Data/Projects/Camelyon17/code/keras_framework/coords/ncrf_cm16/train.txt'
    valid_coord_path = '/media/mak/Data/Projects/Camelyon17/code/keras_framework/coords/ncrf_cm16/valid.txt'
    wsi_path = '/media/mak/mirlproject1/CAMELYON16/TrainingData/normal_tumor'
    label_path = '/media/mak/mirlproject1/CAMELYON16/TrainingData/lesion_masks'

    augmentation = iaa.SomeOf((0, 3), 
                [
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.Noop(),
                    iaa.OneOf([iaa.Affine(rotate=90),
                               iaa.Affine(rotate=180),
                               iaa.Affine(rotate=270)]),
                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                ])
    # Parameters
    train_transform_params = {'image_size': (1024,1024),
                              'batch_size': 1,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 0,
                              'transform': augmentation
                             }

    valid_transform_params = {'image_size': (1024,1024),
                              'batch_size': 16,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 0,
                              'transform': None
                             }
    # Generators
    training_generator = AutomineDataGeneratorCoordFly(wsi_path, label_path, train_coord_path, **train_transform_params)
    validation_generator = AutomineDataGeneratorCoordFly(wsi_path, label_path, valid_coord_path, **valid_transform_params)

    # Enable Test Code
    print (training_generator.__len__())
    print (validation_generator.__len__())
    for X, y in validation_generator:
        print (np.unique(y[0][:,:,1]))
        imshow(normalize_minmax(X[0]), y[0][:,:,1]*255)
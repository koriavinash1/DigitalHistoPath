from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import random

#import imgaug
#from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import numpy as np 
from six.moves import range
import tensorflow as tf
from utils import *
from torchvision import transforms  # noqa
import traceback

# DataLoader Implementation
class DataGeneratorCoordFly(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, wsi_path, mask_path, coord_path, image_size=(1024, 1024), batch_size=32, n_classes=2, n_channels=3,
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

# DataLoader Implementation
class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras. 
    Data format:
        Input and ground truth images are concatenated horizontally next to each other in the following manner.
        | input | class 1 | class 2 |
        Individual images have the same size
    '''
    def __init__(self, data_dir, image_size=(256, 256), batch_size=32, n_classes=2, n_channels=3,
                  shuffle=True, level='L0', transform=None):
        '''
        Parameters:
            data_dir(str): data's root directory
            image_size(tuple - (width,height)): Individual images sizes
            batch_size(int): Batch size
            n_classes(int): Number of classes present in the dataset. As a consequence shape of a single sample image would be (n_classes*batch_size[0],h)
            n_channels(int): number of channels in image
            shuffle(bool): whether to shuffle the data
            level:
            transform:
        '''
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.data_dir = data_dir

        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, '**/*.png')))

        self.image_size = image_size
        self.n_channels = n_channels
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        error = True
        error_index_offset = 0
        while error == True: 
           try:
               'Generate one batch of data'
               # Generate indexes of the batch
               actual_index = error_index_offset + index
               indexes = self.indexes[actual_index*self.batch_size:(actual_index+1)*self.batch_size]
   
               # Find list of IDs
               image_paths_temp = [self.image_paths[k] for k in indexes]
   
               # Generate data
               X, y = self.__data_generation(image_paths_temp)
               error = False
           except Exception as e:
               print(e)
               indexes = self.indexes[actual_index*self.batch_size:(actual_index+1)*self.batch_size]
               image_paths_temp = [self.image_paths[k] for k in indexes]
               print(image_paths_temp)
               traceback.print_tb(e.__traceback__)
               if error_index_offset>=0:
                   error_index_offset=error_index_offset*-1-1
               else:
                   error_index_offset=error_index_offset*-1+1
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _get_one_hot(self, target):
        ''' Encodes a 2d array of labels to a one-hot format
        ''' 
        return (np.arange(self.n_channels) == target[...,None]-1).astype(int)

    def _normalize_image(self, image):
        # Normalize Image
        image = (image - 128.0)/128.0
        # Zero-One Normalization         
        # image = image * (1.0 / 255)  
        # #  ImageNet Standardization
        # imagenet_mean = np.array([0.485, 0.456, 0.406])
        # imagenet_std = np.array([0.229, 0.224, 0.225])
        # image = (image - imagenet_mean) / imagenet_std 
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
  
    def __data_generation(self, image_batch_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y = np.empty((self.batch_size, *self.image_size, self.n_classes))

        # Generate data
        for i, image_mask_path in enumerate(image_batch_list):
           concatenated_image = np.asarray(Image.open(image_mask_path))
           image = concatenated_image[:,:self.image_size[1],:]
           masks = np.zeros(self.image_size)
           for j in range(1,self.n_classes):
               image_width =self.image_size[1]
               try:
                   mask = concatenated_image[:,image_width*j:image_width*(j+1),:]
               except Exception as e:
                   print(f"Encountered Error{e}")
                   print(concatenated_image.shape)
                   print(image_width.shape)
                   raise ValueError
                   
               mask = np.average(mask,axis=2)
               mask = mask.astype('bool')
               np.place(masks,mask,j)
    
           masks = self._get_one_hot(masks)
           if masks.shape[2] !=3:
               import ipdb; ipdb.set_trace()
    
           if self.transform:
               image, mask = self._augmentation(image, mask)
    
           image = self._normalize_image(image)
           X[i,] = image
           y[i,] = masks
        return X, y


if __name__ == '__main__':
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

    #def __init__(self, data_dir, image_size=(256, 256), batch_size=32, n_classes=2, n_channels=3,
    #              shuffle=True, level='L0', transform=None):

    valid_params = [
            os.path.join('..','data','extracted_patches'),
            (768,768),
            2,
            3,
            3,
            True,
            ]
    data_gen = DataGenerator(*valid_params)
    for x in data_gen:
        print(x[1].shape)


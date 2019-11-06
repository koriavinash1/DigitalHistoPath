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
import numpy as np 
from six.moves import range
import openslide
import tensorflow as tf
from torchvision import transforms  # noqa
from torch.utils.data import DataLoader, Dataset
from collections import deque
from math import sin, cos, radians, pi, sqrt


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *

def rotate_list(input_list, N):
    input_list = deque(input_list) 
    input_list.rotate(N) 
    input_list = list(input_list)
    return input_list 

def perturb_coord(x, y, radius=128):
    """
    Perturb the coordinates around the point in circle of radius with uniform distribution
    """
    angle = np.random.uniform(0, 2 * pi)  # in radians
    distance = sqrt(np.random.uniform(0, radius*radius))
    return int(x+distance * cos(angle)), int(y+distance * sin(angle))


class DataGeneratorCoordFly(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, tumor_coord_path, normal_coord_path, image_size=(768, 768), batch_size=32, n_classes=2, n_channels=3,
                  shuffle=True, level=0, samples_per_epoch=None, transform=None):
        'Initialization'
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.tumor_coord_path = tumor_coord_path
        self.normal_coord_path = normal_coord_path
        self.image_size = image_size
        self.n_channels = n_channels
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self.transform = transform
        self.shuffle = shuffle
        self.level = level
        self.tumor_coords = []
        self.normal_coords = []
        t = open(self.tumor_coord_path)
        for line in t:
            pid_path, mask_path, x_center, y_center = line.strip('\n').split(',')[0:4]
            x_center, y_center = int(x_center), int(y_center)
            self.tumor_coords.append((pid_path, mask_path, x_center, y_center))
        t.close()
        n = open(self.normal_coord_path)
        for line in n:
            pid_path, mask_path, x_center, y_center = line.strip('\n').split(',')[0:4]
            x_center, y_center = int(x_center), int(y_center)
            self.normal_coords.append((pid_path, mask_path, x_center, y_center))
        n.close()
        self._num_image = len(self.tumor_coords) + len(self.normal_coords)
        self.tumor_ratio = len(self.tumor_coords)/self._num_image

        if samples_per_epoch is None:
            self.samples_per_epoch = self._num_image
        else:
            self.samples_per_epoch = samples_per_epoch

        self._shuffle_counter = 0        
        self._shuffle_reset_idx = int(np.floor(self._num_image / self.samples_per_epoch))          
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples_per_epoch / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'                        
        try:
            if self.shuffle == True:
                if self._shuffle_counter % self._shuffle_reset_idx == 0:
                    random.shuffle(self.tumor_coords)
                    random.shuffle(self.normal_coords)
            self.tumor_coords = rotate_list(self.tumor_coords, self.samples_per_epoch//2)
            self.normal_coords = rotate_list(self.normal_coords, self.samples_per_epoch//2)
            self._shuffle_counter += 1
        except Exception as error:
            print(error)
            import ipdb; ipdb.set_trace()


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
        norm_batch_size = self.batch_size//2
        tumor_batch_size = self.batch_size - self.batch_size//2

        for i in range(self.batch_size):

            #print((index//2)*norm_batch_size+i,index*tumor_batch_size//2+i,len(self.normal_coords),len(self.tumor_coords))
            if i < self.batch_size//2:
                pid_path, mask_path, x_center, y_center = self.normal_coords[int(index*norm_batch_size+i)%len(self.normal_coords)]
            else:        
                pid_path, mask_path, x_center, y_center = self.tumor_coords[int(index*tumor_batch_size+i)%len(self.tumor_coords)]        

            # Generate data
            x_center, y_center = perturb_coord(x_center, y_center)
            norm_batch_size = self.batch_size//2
            tumor_batch_size = self.batch_size - self.batch_size//2
            x_top_left = int(int(x_center) - self.image_size[0] / 2)
            y_top_left = int(int(y_center) - self.image_size[1] / 2)
            try:
                image_opslide = openslide.OpenSlide(pid_path)
            except Exception as e: 
                print(100*('-'))
                print(pid_path)
                print(e)
                raise ValueError


            x_max_dim,y_max_dim = image_opslide.level_dimensions[self.level]

            if x_top_left < 0:
                x_top_left = 0
            elif x_top_left>x_max_dim - self.image_size[0]:
                x_top_left = x_max_dim - self.image_size[0]
            
            if y_top_left < 0:
                y_top_left = 0
            elif y_top_left>y_max_dim - self.image_size[1]:
                y_top_left = y_max_dim - self.image_size[1]

            image = image_opslide.read_region(
                (x_top_left, y_top_left), self.level,
                (self.image_size[0], self.image_size[1])).convert('RGB')
            if mask_path !='0':
                try:
                    mask_opslide = openslide.OpenSlide(mask_path)
                except Exception as e: 
                    print(100*('-'))
                    print(pid_path)
                    print(e)
                    raise ValueError
                mask = mask_opslide.read_region(
                    (x_top_left, y_top_left), self.level,
                    (self.image_size[0], self.image_size[1])).convert('L')
            else:
                mask = np.zeros((self.image_size[0], self.image_size[1]))
            image = np.asarray(image)
            mask =  np.asarray(mask,dtype=np.bool)
            mask =  np.uint8(mask)
            mask = self._get_one_hot(mask)
            if self.transform:
                image, mask = self._augmentation(image, mask)
            image = self._normalize_image(image)
            X[i,] = image
            y[i,] = mask
        return X, y

if __name__ == '__main__':


    from utils import imsave
 
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
    train_transform_params = {'image_size': (768,768),
                          'batch_size': 4,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          # 'samples_per_epoch': 60000,
                          'samples_per_epoch': 4000,
                          'transform': augmentation
                         }

    valid_transform_params = {'image_size': (256, 256),
                          'batch_size': 8,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          'samples_per_epoch': 30000,                        
                          'transform': None
                         }
    dir_path = '/media/brats/mirlproject2/haranrk/paip-2019/data/raw-data'
    train_tumor_coord_path = os.path.join(dir_path, 'train_dummy_tumor.txt')
    train_normal_coord_path = os.path.join(dir_path, 'train_dummy_normal.txt')

    #Generators
    training_generator = DataGeneratorCoordFly(train_tumor_coord_path, train_normal_coord_path, **train_transform_params)
    print (training_generator.__len__())
    # # Enable Test Code
    for i,(X, y) in enumerate(training_generator):
        print('Iter %d'%(i))
        print(X.shape, y.shape)
        #imsave(X[0],y[0,:,:,1], out='ref2_%d.png'%(i))
        imsave(X[0], y[0][:,:,1], X[1], y[1][:,:,1],X[2], y[2][:,:,1], X[3], y[3][:,:,1], out='ref_%d.png' % (i))        
        #import ipdb; ipdb.set_trace()


    valid_tumor_coord_path = os.path.join(dir_path, 'valid_tumor.txt')
    valid_normal_coord_path = os.path.join(dir_path, 'valid_normal.txt')
    validation_generator = DataGeneratorCoordFly(valid_tumor_coord_path, valid_normal_coord_path, **valid_transform_params)
    print (validation_generator.__len__())
    # Enable Test Code

    # for X, y in validation_generator:
    #     pass
        # imshow(normalize_minmax(X[0]), y[0][:,:,1], normalize_minmax(X[1]), y[1][:,:,1], \
        #     normalize_minmax(X[2]), y[2][:,:,1], normalize_minmax(X[3]), y[3][:,:,1])  
    import time        
    start_time = time.time()    
    for i, X in enumerate(validation_generator):
        elapsed_time = time.time() - start_time
        start_time = time.time()    
        print (i, "Elapsed Time", np.round(elapsed_time, decimals=2), "seconds")
        pass

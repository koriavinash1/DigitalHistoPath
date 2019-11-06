from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import timeit

# Random Seeds
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

from tensorflow.keras import backend as K
from imgaug import augmenters as iaa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
import argparse  

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from dataloader.training_data_loader import DataGeneratorCoordFly
from helpers.utils import *
from models.seg_models import unet_densenet121
from tensorflow.keras.utils import multi_gpu_model

parser = argparse.ArgumentParser(description='Training DFCN')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')


def main(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path,
         model_path, use_pretrained_model_weights_path=None, restore_model=False,
        initial_epoch=0, n_Epochs = 50):
    #this block enables GPU enabled multiprocessing 
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    session =tf.Session(config=core_config) 
    K.set_session(session)

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
    train_transform_params = {'image_size': (256, 256),
                          'batch_size': 16,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          'samples_per_epoch': 64000,
                          'transform': augmentation
                         }

    valid_transform_params = {'image_size': (256, 256),
                          'batch_size': 32,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          'samples_per_epoch': 16000,                        
                          'transform': None
                         }
    # Generators
    training_generator = DataGeneratorCoordFly(train_tumor_coord_path, train_normal_coord_path, **train_transform_params)
    validation_generator = DataGeneratorCoordFly(valid_tumor_coord_path, valid_normal_coord_path, **valid_transform_params)
    print ("No. of training and validation batches are:", training_generator.__len__(), validation_generator.__len__())

    # Model Configuration

    logdir_path = os.path.join(model_path, 'tb_logs')
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)

    # Callback Configuration
    tbCallback = TensorBoard(log_dir=logdir_path, histogram_freq=0, write_graph=True, write_images=True)
    model_checkpoint = ModelCheckpoint(os.path.join(model_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', 
                                        save_best_only=False, save_weights_only=True, mode='min')

    if (not use_pretrained_model_weights_path) and (initial_epoch == 0):
        print ('Model Starting with pretrained imagenet weights')
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))        
        model = unet_densenet121((None, None), weights='imagenet')
        try:
            # pass
            model = multi_gpu_model(model, gpus=len(args.GPU.split(',')), cpu_relocation=True)
            print("Training using multiple GPUs..")
        except:
            print("Training using single GPU or CPU..")   

        model.compile(loss=softmax_dice_focal_loss,
                        optimizer=Adam(lr=3e-4, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, 
                        metrics.binary_accuracy, metrics.categorical_crossentropy])

        model.fit_generator(generator=training_generator,
                                epochs=6, verbose=1,
                                validation_data=validation_generator,
                                callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=0)

        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]))
        for l in model.layers:
            l.trainable = True
        model.compile(loss=softmax_dice_focal_loss,
                        optimizer=Adam(lr=5e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1,
                        metrics.binary_accuracy, metrics.categorical_crossentropy])

        model.fit_generator(generator=training_generator,
                                epochs=n_Epochs, verbose=1,
                                validation_data=validation_generator,
                                callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=6)

    elif use_pretrained_model_weights_path is not None:
        # Run Model
        print ("Loadng model from pretrained weights")
        model = unet_densenet121((None, None), weights=None)
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]))        
        if os.path.exists(use_pretrained_model_weights_path):
            model.load_weights(use_pretrained_model_weights_path)
            print("Loaded pretrained_model_weights from disk")
        else:
            print("Unable to load pre-trained model weights")    
        try:
            # pass
            model = multi_gpu_model(model, gpus=len(args.GPU.split(',')), cpu_relocation=True)
            print("Training using multiple GPUs..")
        except:
            print("Training using single GPU or CPU..")   
        model.compile(loss=softmax_dice_focal_loss,
                        optimizer=Adam(lr=5e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1,
                        metrics.binary_accuracy, metrics.categorical_crossentropy])
        model.fit_generator(generator=training_generator,
                                epochs=n_Epochs, verbose=1,
                                validation_data=validation_generator,
                                callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=initial_epoch)

    elif restore_model:
        model = load_model(os.path.join(model_path, 'model.xxx.h5'),
                        custom_objects={'dice_coef_rounded_ch0': dice_coef_rounded_ch0, 
                        'dice_coef_rounded_ch1': dice_coef_rounded_ch1,
                        'softmax_dice_focal_loss':softmax_dice_focal_loss
                        })
        try:
            # pass
            model = multi_gpu_model(model, gpus=len(args.GPU.split(',')), cpu_relocation=True)
            print("Training using multiple GPUs..")
        except:
            print("Training using single GPU or CPU..")   
        model.summary()       
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]))                
        model.fit_generator(generator=training_generator,
                                epochs=n_Epochs-initial_epoch, verbose=2,
                                validation_data=validation_generator,
                                callbacks=[lrSchedule, model_checkpoint, tbCallback],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=initial_epoch)
    del model
    del model_checkpoint        
    K.clear_session()


if __name__ == '__main__':
    args = parser.parse_args()
    t0 = timeit.default_timer()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    print ("GPU Availability: ", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    # Training Data Configuration
    fold_list = [0,1,2]
    for fold_no in fold_list:    
        dir_path = '/media/mak/Data/Projects/Camelyon17/code/keras_framework/patch_coords/cm17_16_train_test_ncrf_points_fold_{}'.format(fold_no)
        train_tumor_coord_path = os.path.join(dir_path, 'train_tumor.txt')
        train_normal_coord_path = os.path.join(dir_path, 'train_normal.txt')
        valid_tumor_coord_path = os.path.join(dir_path, 'valid_tumor.txt')
        valid_normal_coord_path = os.path.join(dir_path, 'valid_normal.txt')

        # Model Path
        model_path = '/media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/CM17/IncpResV2_UNET_CM17_RANDOM_16_NCRF_BCE_DICE_fold_{}'.format(fold_no)    
        pretrained_model_path = None        
        main(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path,\
         model_path, use_pretrained_model_weights_path=pretrained_model_path, restore_model=False, initial_epoch=0, n_Epochs = 50)
        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))
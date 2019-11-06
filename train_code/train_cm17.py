from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import numpy as np
import random
import tensorflow as tf

# Random Seeds
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

from tensorflow.keras import backend as K
from data_loader import DataGenerator
from imgaug import augmenters as iaa
from models import unet_densenet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from utils import dice_coef_rounded_ch0, dice_coef_rounded_ch1, softmax_dice_loss, schedule_steps



def main(train_data_dir, valid_data_dir, model_path): 
    #this block enables GPU enabled multiprocessing 
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    session =tf.Session(config=core_config) 
    K.set_session(session)

    augmentation = iaa.SomeOf((0, 3), 
                [
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.OneOf([iaa.Affine(rotate=90),
                               iaa.Affine(rotate=180),
                               iaa.Affine(rotate=270)]),
                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                ])
    # Parameters
    train_transform_params = {'image_size': (256,256),
                              'batch_size': 16,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 'L0',
                              'transform': augmentation
                             }

    valid_transform_params = {'image_size': (256,256),
                              'batch_size': 16,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 'L0',
                              'transform': None
                             }
    # Generators
    training_generator = DataGenerator(train_data_dir, **train_transform_params)
    validation_generator = DataGenerator(valid_data_dir, **valid_transform_params)
    print (training_generator.__len__(), validation_generator.__len__())

    # Model Configuration
    n_Epochs1 = 6
    lrSchedule1 = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))
    n_Epochs2 = 90
    lrSchedule2 = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 90)]))


    logdir_path = os.path.join(model_path, 'tb_logs')
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)

    # Callback Configuration
    tbCallback = TensorBoard(log_dir=logdir_path, histogram_freq=0, write_graph=True, write_images=True)

    # Run Model
    # First Level training with fixed pretrained imagenet weights for encoder and only updating decoder weights
    model = unet_densenet121((None, None), weights='imagenet')
    model.compile(loss=softmax_dice_loss,
                    optimizer=Adam(lr=3e-4, amsgrad=True),
                    metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])

    model_checkpoint1 = ModelCheckpoint(os.path.join(model_path, 'Model_Stage1.h5'), monitor='val_loss', verbose=1,
                                        save_best_only=True, save_weights_only=True, mode='min')

    model.fit_generator(generator=training_generator,
                            epochs=n_Epochs1, verbose=1,
                            validation_data=validation_generator,
                            callbacks=[lrSchedule1, model_checkpoint1],
                            use_multiprocessing=True,
                            workers=6)

    # Second Level training with updates for weights of encoder and decoder 
    # Make the layer weights trainable
    for l in model.layers:
        l.trainable = True
        
    model.compile(loss=softmax_dice_loss,
                    optimizer=Adam(lr=5e-6, amsgrad=True),
                    metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])

    model_checkpoint2 = ModelCheckpoint(os.path.join(model_path, 'Model_Stage2.h5'), monitor='val_loss', verbose=1,
                                        save_best_only=True, save_weights_only=True, mode='min')

    model.fit_generator(generator=training_generator,
                            epochs=n_Epochs2, verbose=1,
                            validation_data=validation_generator,
                            callbacks=[lrSchedule2, model_checkpoint2, tbCallback],
                            use_multiprocessing=True,
                            workers=6)
    del model
    del model_checkpoint
    K.clear_session()

if __name__ == '__main__':
    # Training Data Configuration
    # Data Path
    train_data_dir = "/media/mak/Data/Projects/Camelyon17/dataset/cm17_patches/train"
    valid_data_dir = "/media/mak/Data/Projects/Camelyon17/dataset/cm17_patches/val"
    # Model Path
    model_path = '../../saved_models/keras_models/segmentation/unet_densenet121_imagenet_pretrained_L0_'+datetime.now().strftime("%Y%m%d-%H%M%S")

    main(train_data_dir, valid_data_dir, model_path)
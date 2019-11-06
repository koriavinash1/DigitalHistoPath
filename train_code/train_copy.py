from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
from data_loader import DataGeneratorCoordFly
from imgaug import augmenters as iaa
from models import unet_densenet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from utils import dice_coef_rounded_ch0, dice_coef_rounded_ch1, softmax_dice_loss, schedule_steps



def main(wsi_path, mask_path, train_coord_path, valid_coord_path, model_path): 
    #this block enables GPU enabled multiprocessing 
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    session =tf.Session(config=core_config) 
    K.set_session(session)

    augmentation = iaa.SomeOf((0, 4), 
                [
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.OneOf([iaa.Affine(rotate=90),
                               iaa.Affine(rotate=180),
                               iaa.Affine(rotate=270)]),
                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                ])
    # Parameters
    train_transform_params = {'image_size': (512,512),
                              'batch_size': 4,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 0,
                              'transform': augmentation
                             }

    valid_transform_params = {'image_size': (512,512),
                              'batch_size': 16,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 0,
                              'transform': None
                             }
    # Generators
    training_generator = DataGeneratorCoordFly(wsi_path, mask_path, train_coord_path, **train_transform_params)
    validation_generator = DataGeneratorCoordFly(wsi_path, mask_path, valid_coord_path, **valid_transform_params)
    print ("No. of training and validation batches are:", training_generator.__len__(), validation_generator.__len__())

    # Model Configuration
    n_Epochs1 = 2
    lrSchedule1 = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 1), (3e-4, 2)]))
    n_Epochs2 = 10
    lrSchedule2 = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(2e-4, 5), (1e-4, 10)]))


    logdir_path = os.path.join(model_path, 'tb_logs')
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)

    # Callback Configuration
    tbCallback = TensorBoard(log_dir=logdir_path, histogram_freq=0, write_graph=True, write_images=True)

    # Run Model
    # First Level training with fixed pretrained imagenet weights for encoder and only updating decoder weights
    model = unet_densenet121((None, None), weights='imagenet')
    model.summary()
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
                    optimizer=Adam(lr=2e-4, amsgrad=True),
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
    del model_checkpoint1
    del model_checkpoint2

    K.clear_session()


if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # Training Data Configuration
    train_coord_path = '/media/mak/Data/Projects/Camelyon17/code/keras_framework/coords/ncrf_cm16/train.txt'
    valid_coord_path = '/media/mak/Data/Projects/Camelyon17/code/keras_framework/coords/ncrf_cm16/valid.txt'
    wsi_path = '/media/mak/mirlproject1/CAMELYON16/TrainingData/normal_tumor'
    mask_path = '/media/mak/mirlproject1/CAMELYON16/TrainingData/lesion_masks'

    # Model Path
    model_path = '../../../saved_models/keras_models/segmentation/CM16/unet_densenet121_imagenet_pretrained_L0_'+datetime.now().strftime("%Y%m%d-%H%M%S")
    main(wsi_path, mask_path, train_coord_path, valid_coord_path, model_path)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
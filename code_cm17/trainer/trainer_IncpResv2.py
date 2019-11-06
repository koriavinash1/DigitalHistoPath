from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
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
from models.seg_models import unet_densenet121, get_inception_resnet_v2_unet_softmax
from tensorflow.keras.utils import multi_gpu_model

parser = argparse.ArgumentParser(description='Training DFCN')
parser.add_argument('--GPU', default='1', type=str, help='which GPU to use'
                    ', default 1')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path, model_path, use_pretrained_model_weights=False, restore_model=False,
    initial_epoch=0): 
    #this block enables GPU enabled multiprocessing 
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    # core_config.gpu_options.per_process_gpu_memory_fraction=0.6
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
                          'samples_per_epoch': None,
                          # 'samples_per_epoch': 8001,
                          'transform': augmentation
                         }

    valid_transform_params = {'image_size': (256, 256),
                          'batch_size': 16,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          'samples_per_epoch': None,                        
                          # 'samples_per_epoch': 8001,
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

    if (not use_pretrained_model_weights) and (initial_epoch == 0):
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))        
        model = get_inception_resnet_v2_unet_softmax((None, None), weights='imagenet')
        try:
            # pass
            model = multi_gpu_model(model, gpus=len(args.GPU.split(',')), cpu_relocation=True)
            print("Training using multiple GPUs..")
        except:
            print("Training using single GPU or CPU..")   

        model.compile(loss=softmax_dice_loss,
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
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=5e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1,
                        metrics.binary_accuracy, metrics.categorical_crossentropy])

        model.summary()
        model.fit_generator(generator=training_generator,
                                epochs=100, verbose=1,
                                validation_data=validation_generator,
                                callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=6)

        del model
        del model_checkpoint
        K.clear_session()

    elif use_pretrained_model_weights:
        # Run Model
        print ("Loadng model from pretrained weights")
        n_Epochs = 100
        model = get_inception_resnet_v2_unet_softmax((None, None), weights=None)
        # lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]),verbose=1)        
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(2e-6, 2), (1e-6, 10), (5e-7, 15), (2e-5, 80), (1e-5, 100)]),verbose=1)        
        pretrained_model_weights_path = os.path.join(model_path, '..','pre-model.18-0.18.h5')
        # pretrained_model_weights_path = glob.glob(os.path.join(model_path,'sel-model*.h5'))[0]
        print("Starting trainig from model stored in %s" % pretrained_model_weights_path)
        if os.path.exists(pretrained_model_weights_path):
            model.load_weights(pretrained_model_weights_path)
            print("Loaded pretrained_model_weights from disk")
        else:
            print("Unable to load pre-trained model weights")    
        print("Starting trainig from model stored in %s" % pretrained_model_weights_path)
        # try:
            # # pass
            # model = multi_gpu_model(model, gpus=len(args.GPU.split(',')), cpu_relocation=True)
            # print("Training using multiple GPUs..")
        # except:
            # print("Training using single GPU or CPU..")   
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=1e-5, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1,
                        metrics.binary_accuracy, metrics.categorical_crossentropy])

        model.fit_generator(generator=training_generator,
                                epochs=n_Epochs, verbose=1,
                                validation_data=validation_generator,
                                callbacks=[tbCallback, lrSchedule,model_checkpoint],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=initial_epoch)

    elif restore_model and initial_epoch:
        print ('Restoring Model', 'Number of GPUs:', len(args.GPU.split(',')))
        n_Epochs = 100
        model = load_model(os.path.join(model_path, 'model.xxx.h5'),
                        custom_objects={'dice_coef_rounded_ch0': dice_coef_rounded_ch0, 
                        'dice_coef_rounded_ch1': dice_coef_rounded_ch1,
                        'softmax_dice_loss':softmax_dice_loss
                        })
        model = multi_gpu_model(model, gpus=len(args.GPU.split(',')), cpu_relocation=True)
        print("Training using multiple GPUs..")
        # try:
        #     # pass
        #     model = multi_gpu_model(model, gpus=len(args.GPU), cpu_relocation=True)
        #     print("Training using multiple GPUs..")
        # except:
        #     print("Training using single GPU or CPU..")   
        model.summary()       
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]))                
        model.fit_generator(generator=training_generator,
                                epochs=n_Epochs-initial_epoch, verbose=1,
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
    
    print ("GPU Availability: ", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    # Training Data Configuration
    kfold_k = 5
    #for fold in range(kfold_k):
    fold = 2
    model_id = 'incep_viable_200k'
    print('Starting %d fold %d'%(kfold_k, fold))

    #Normal training case
    tumor_type = 'viable'
    coord_set = 'patch_coords_200k'
    dir_path = '../../data/raw-data/%s/%dfold_%d'%(coord_set,kfold_k,fold)
    path_gen = lambda mode,tumor,coord: os.path.join(dir_path,'%s_%s_%s.txt'%(mode,tumor,coord))
    train_tumor_coord_path = path_gen('training',tumor_type,'tumor')
    train_normal_coord_path = path_gen('training',tumor_type,'normal')

    # Mining case
    # train_normal_coord_path = '../../results/saved_models/%s/%dfold_%d/mined_points/normal.txt'%(model_id,kfold_k, fold)
    # train_tumor_coord_path = '../../results/saved_models/%s/%dfold_%d/mined_points/tumor.txt'%(model_id,kfold_k, fold)


    valid_tumor_coord_path = path_gen('validation',tumor_type,'tumor')
    valid_normal_coord_path = path_gen('validation',tumor_type,'normal')

    print(train_normal_coord_path)
    print(train_tumor_coord_path)
    print(valid_normal_coord_path)
    print(valid_tumor_coord_path)

    # Model Path
    #model_path = '/media/balaji/Kori/histopath/code_base/model_weights'
    model_path = '../../results/saved_models/%s/%dfold_%d'%(model_id,kfold_k, fold)
    os.makedirs(model_path, exist_ok=True)
    main(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path,\
     model_path, use_pretrained_model_weights=True, restore_model=False, initial_epoch=0)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))

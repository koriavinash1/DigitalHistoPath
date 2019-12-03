from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys,glob
from datetime import datetime
import numpy as np
import random
import argparse  

import tensorflow as tf
from tensorflow.keras import backend as K
from imgaug import augmenters as iaa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
# Random Seeds
np.random.seed(0)
random.seed(0)
tf.compat.v1.set_random_seed(0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from dataloader.training_data_loader import DataGeneratorCoordFly
from helpers.utils import *
from models.seg_models import unet_densenet121, get_inception_resnet_v2_unet_softmax
from models.deeplabv3p_original import Deeplabv3

def train(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path,
         model_path, use_pretrained_model_weights_path=None, restore_model=False,
         initial_epoch=0, n_Epochs=50):
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
                          'batch_size': args.train_bz,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          'samples_per_epoch': None,
                          'transform': augmentation
                         }

    valid_transform_params = {'image_size': (256, 256),
                          'batch_size': args.valid_bz,
                          'n_classes': 2,
                          'n_channels': 3,
                          'shuffle': True,
                          'level': 0,
                          'samples_per_epoch': None,                        
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
    tbCallback = TensorBoard(log_dir=logdir_path, histogram_freq=0, write_graph=True, write_images=False)
    model_checkpoint = ModelCheckpoint(os.path.join(model_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', 
                                        save_best_only=False, save_weights_only=True, mode='min')

    if (not use_pretrained_model_weights_path) and (initial_epoch == 0):
        if args.model == 'deeplab':
            print ('Model Starting with Pascal-VOC weights')
            model = Deeplabv3(input_shape=(256, 256, 3), weights='pascal_voc', classes=2,  backbone='xception', OS=16, activation='softmax')
            # model.summary()
            lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]))
            model.compile(loss=softmax_dice_loss,
                            optimizer=Adam(lr=5e-6, amsgrad=True),
                            metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1,
                            metrics.binary_accuracy, metrics.categorical_crossentropy])

            model.fit_generator(generator=training_generator,
                                    epochs=n_Epochs, verbose=1,
                                    validation_data=validation_generator,
                                    callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                    use_multiprocessing=True,
                                    workers=6,
                                    initial_epoch=0)
        else:
            print('%s Model Starting with pretrained imagenet weights' % args.model)
            if args.model == 'inception':
                model = get_inception_resnet_v2_unet_softmax((None, None), weights='imagenet')
            elif args.model == 'densenet':
                model = unet_densenet121((None, None), weights='imagenet')
            else:
                print("Model type not known")

            model.compile(loss=softmax_dice_loss,
                            optimizer=Adam(lr=3e-4, amsgrad=True),
                            metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, 
                            metrics.binary_accuracy, metrics.categorical_crossentropy])

            lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))        
            model.fit_generator(generator=training_generator,
                                    epochs=2, verbose=1,
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

            model.fit_generator(generator=training_generator,
                                    epochs=n_Epochs, verbose=1,
                                    validation_data=validation_generator,
                                    callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                    use_multiprocessing=True,
                                    workers=6,
                                    initial_epoch=2)

    elif use_pretrained_model_weights_path is not None:
        # Run Model
        print ("Loadng model from pretrained weights")
        if args.model == 'densenet':
            model = unet_densenet121((None, None), weights=None)
        elif args.model == 'inception':
            model = get_inception_resnet_v2_unet_softmax((None, None), weights=None)
        elif args.model == 'deeplab':
            model = Deeplabv3(input_shape=(256, 256, 3), weights='pascal_voc', classes=2,  backbone='xception', OS=16, activation='softmax')        
        else:
            print("Model type not known")

        if os.path.exists(use_pretrained_model_weights_path):
            model.load_weights(use_pretrained_model_weights_path)
            print("Loaded pretrained_model_weights from disk %s " % (use_pretrained_model_weights_path))
        else:
            print("Unable to load pre-trained model weights")    

        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=5e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1,
                        metrics.binary_accuracy, metrics.categorical_crossentropy])

        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 100)]))        
        model.fit_generator(generator=training_generator,
                                epochs=n_Epochs, verbose=1,
                                validation_data=validation_generator,
                                callbacks=[lrSchedule, tbCallback, model_checkpoint],
                                use_multiprocessing=True,
                                workers=6,
                                initial_epoch=initial_epoch)

    del model
    del model_checkpoint        
    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training DFCN')
    parser.add_argument('model',type=str, choices=['densenet','inception','deeplab'], help="model to train")
    parser.add_argument('fold',type=int,help="Which fold to train on")
    parser.add_argument('--GPU', default='0', type=str, help='which GPU to use. default 0')
    parser.add_argument('--train_bz', default='16', type=int, help='batch size for training')
    parser.add_argument('--valid_bz', default='32', type=int, help='batch size for vallation')
    parser.add_argument('--override', action='store_true', help='Whether to override the directory if the directory already exists')
    parser.add_argument('-r','--resume', action='store_true', help='Resume training if previous training was found')

    args = parser.parse_args()

    
    # Training Data Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    kfold_k = 5
    fold = args.fold
    model_id = '%s_200k' % (args.model)
    print("-"*10)
    print("Training a %s model | %dfold %d | batch size %d,%d | GPU %s" %( args.model, kfold_k, args.fold, args.train_bz, args.valid_bz, args.GPU))
    print("Saving model to %s" % model_id)
    print("-"*10)

    #Normal training case
    tumor_type = 'viable'
    coord_set = 'patch_coords_200k'
    dir_path = '../../data/raw-data/%s/%dfold_%d'%(coord_set,kfold_k,fold)
    path_gen = lambda mode,tumor,coord: os.path.join(dir_path,'%s_%s_%s.txt'%(mode,tumor,coord))
    train_tumor_coord_path = path_gen('training',tumor_type,'tumor')
    train_normal_coord_path = path_gen('training',tumor_type,'normal')

    valid_tumor_coord_path = path_gen('validation',tumor_type,'tumor')
    valid_normal_coord_path = path_gen('validation',tumor_type,'normal')

    print(train_normal_coord_path)
    print(train_tumor_coord_path)
    print(valid_normal_coord_path)
    print(valid_tumor_coord_path)

    # Model Path
    model_path = '../../results/saved_models/%s/%dfold_%d'%(model_id,kfold_k,fold)
    if args.resume:
        model_paths = glob.glob(os.path.join(model_path,"model*"))
        model_paths.sort()
        pretrained_model_path = model_paths[-1]
        initial_epoch = int(os.path.basename(pretrained_model_path).split('-')[0].split('.')[1])
        print("Continue with model %s from epoch %d ?"% (pretrained_model_path, initial_epoch+1))
        input()
        train(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path,\
         model_path, use_pretrained_model_weights_path=pretrained_model_path, initial_epoch=initial_epoch, n_Epochs=100)
    else:
        try:
            os.makedirs(model_path)
        except FileExistsError:
            if os.listdir(model_path) != []:
                print("Out folder exists and is non-empty, continue?")
                print(model_path)
                input()
        pretrained_model_path = None
        initial_epoch = 0
        train(args, train_tumor_coord_path, train_normal_coord_path, valid_tumor_coord_path, valid_normal_coord_path,\
         model_path, use_pretrained_model_weights_path=pretrained_model_path, initial_epoch=initial_epoch, n_Epochs=100)

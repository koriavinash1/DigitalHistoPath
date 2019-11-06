from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import json
import glob
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
from data_loader import DataGenerator
#from imgaug import augmenters as iaa
from models import unet_densenet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from utils import dice_coef_rounded_ch0, dice_coef_rounded_ch1, softmax_dice_loss, schedule_steps
from tensorflow.keras.callbacks import Callback
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class WeightSaver(Callback):
    def __init__(self, N, prefix, lock_path, add_info, start_batch=0):
        ''' 
        params:
            N: Batch interval between which to save
            prefix: dir prefix to be added to the save file
        '''
        self.N = N
        self.batch = start_batch
        self.prefix = prefix
        self.lock_path = lock_path
        self.add_info = add_info
    
    def on_batch_end(self,batch,logs={}):
        if self.batch % self.N ==0:
            name = f'{self.prefix}_{self.batch}.h5'
            self.model.save_weights(name)
            self.add_info['batch'] = self.batch
            with open(self.lock_path,'w') as json_file:
                json.dump(self.add_info,json_file)
        self.batch +=1

class EarlyStoppingByBatch(Callback):
    def __init__(self, n_batches):
        '''
        params:
            n_batches: Batches after which to stop training
        '''
        self.n_batches = n_batches

    def on_batch_end(self,batch,logs={}):
        if batch>self.n_batches:
            self.model.stop_training = True

def main(model_path): 
    #this block enables GPU enabled multiprocessing 
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    core_config.gpu_options.per_process_gpu_memory_fraction=0.5
    session =tf.Session(config=core_config) 
    print(f"Creating tf session")
    K.set_session(session)

    # Parameters
    #def __init__(self, data_dir, image_size=(256, 256), batch_size=32, n_classes=2, n_channels=3,
    #              shuffle=True, level='L0', transform=None):

    valid_params = [
            os.path.join('..','data','extracted_patches', 'train'),
            (768,768),
            2,
            3,
            3,
            True,
            ]
    data_generator = DataGenerator(*valid_params)

    # Generators
    print ("No. of training batches are:", data_generator.__len__() )

    # Model Configuration
    n_Epochs1 = 2
    lrSchedule1 = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 1), (3e-4, 2)]))
    n_Epochs2 = 10
    lrSchedule2 = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(2e-4, 5), (1e-4, 10)]))

    # Callback Configuration
    logdir_path = os.path.join(model_path, 'tb_logs')
    lock_path = os.path.join(model_path,'lock.json')

    tbCallback = TensorBoard(log_dir=logdir_path, histogram_freq=0, write_graph=True, write_images=True)
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)

    
    n_class=3
    model = unet_densenet121((None, None),n_class,weights='imagenet')
    model.summary()
    model.compile(loss=softmax_dice_loss,
                    optimizer=Adam(lr=3e-4, amsgrad=True),
                    metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])

    if os.path.isfile(lock_path):
        with open(lock_path,'r') as json_file:
            current_status = json.load(json_file)
        
        model_paths = glob.glob(os.path.join(model_path,'*.h5'))
        model_paths.sort()
        print(f"Loading model weights stored in {model_paths[-1]}")
        model.load_weights(model_paths[-1])

        #TODO Remove hardcoding
        batches_per_epoch = 50000
        batches_left_in_last_epoch = batches_per_epoch - current_status['batch'] % batches_per_epoch
        num_epochs_left = n_Epochs2+n_Epochs1 - current_status['batch'] // batches_per_epoch -1
        print(f"Completing the last {batches_left_in_last_epoch} batches in the interrupted epoch")
        early_stopping = EarlyStoppingByBatch(batches_left_in_last_epoch)
        model_batch_checkpoint2 = WeightSaver(50,os.path.join(model_path,'Model_Stage2_batch'),lock_path,{'stage': 2}, start_batch=current_status['batch'])
        model_checkpoint2 = ModelCheckpoint(os.path.join(model_path, 'Model_Stage2.h5'), monitor='val_loss', verbose=1,
                                            save_best_only=True, save_weights_only=True, mode='min')
        model.fit_generator(generator=data_generator,
                                epochs=1, verbose=1,
                                validation_data=data_generator,
                                callbacks=[lrSchedule2, model_checkpoint2, tbCallback,model_batch_checkpoint2, early_stopping],
                                use_multiprocessing=True,
                                workers=6)
        print(f'Resuming training from {current_status["batch"]//batches_per_epoch}/{n_Epochs1+n_Epochs2}')
        model.fit_generator(generator=data_generator,
                                epochs=num_epochs_left, verbose=1,
                                validation_data=data_generator,
                                callbacks=[lrSchedule2, model_checkpoint2, tbCallback,model_batch_checkpoint2 ],
                                use_multiprocessing=True,
                                workers=6)
    else:
        # Run Model
        # First Level training with fixed pretrained imagenet weights for encoder and only updating decoder weights
        print(f"First Level: Setting up model")
        model_checkpoint1 = ModelCheckpoint(os.path.join(model_path, 'Model_Stage1.h5'), monitor='val_loss', verbose=1,
                                            save_best_only=True, save_weights_only=True, mode='min')
        model_batch_checkpoint1 = WeightSaver(50,os.path.join(model_path,'Model_Stage1_batch'),lock_path,{'stage': 1} )

        print(f"First Level: Training model")
        model.fit_generator(generator=data_generator,
                                epochs=n_Epochs1, verbose=1,
                                validation_data=data_generator,
                                callbacks=[lrSchedule1, model_checkpoint1, model_batch_checkpoint1],
                                use_multiprocessing=True,
                                workers=6)

        # Second Level training with updates for weights of encoder and decoder 
        # Make the layer weights trainable
        for l in model.layers:
            l.trainable = True        
        print(f"Second Level: Setting up model")
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=2e-4, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])
        model_checkpoint2 = ModelCheckpoint(os.path.join(model_path, 'Model_Stage2.h5'), monitor='val_loss', verbose=1,
                                            save_best_only=True, save_weights_only=True, mode='min')
        model_batch_checkpoint2 = WeightSaver(50,os.path.join(model_path,'Model_Stage2_batch'),lock_path,{'stage': 2} )
        print(f"Second Level: Training model")
        model.fit_generator(generator=data_generator,
                                epochs=n_Epochs2, verbose=1,
                                validation_data=data_generator,
                                callbacks=[lrSchedule2, model_checkpoint2, tbCallback],
                                use_multiprocessing=True,
                                workers=6)

    K.clear_session()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    # Model Path
    results_path = os.path.join('..','results')
    saved_models_path = os.path.join(results_path, 'saved_models')

    train_jobs = os.listdir(saved_models_path)
    print(train_jobs)

    #unet_densenet121_imagenet_pretrained_L0_
    train_job_id = 'unet'

    if train_jobs != [] and os.path.isfile(os.path.join(saved_models_path,train_jobs[-1], 'lock.json')):
            print(f'Lock file detected in job {train_jobs[-1]}. Attempting to resume training')
            model_path = os.path.join(saved_models_path,train_jobs[-1])
    else:
        print(f'No previous training for job: {train_job_id}. Starting new job.\n')
        model_path = os.path.join(results_path,'saved_models',train_job_id +datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(model_path, exist_ok=True)
    #    open(os.path.join(model_path,'lock.json'),'a').close()

    main(model_path)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    os.rename(os.path.join(model_path,'lock.json'),os.path.join(model_path,'finished.json'))

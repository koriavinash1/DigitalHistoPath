import numpy as np
from data_loader import DataGenerator
from models import unet_densenet121
from utils import imshow,imsave
import os
import glob
import openslide 
from skimage import io
import cv2
import time
import tensorflow as tf
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def load_model(model_path,n_class=3, use_gpu=True):
    '''
    Loads the given model
    '''
    if use_gpu:
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True 
        core_config.gpu_options.per_process_gpu_memory_fraction=0.4
        session =tf.Session(config=core_config) 
        print(f"Creating tf session")
        K.set_session(session)

    model = unet_densenet121((None, None),n_class,weights=None)
    model_paths = glob.glob(os.path.join(model_path,'*.h5'))
    model_paths.sort()
    print(f"Loading model weights stored in {model_paths[-1]}")
    model.load_weights(model_paths[-1])
    return model

def load_dataset(generator_params):
    data_generator = DataGenerator(*generator_params)
    return data_generator

def predict_from_patches(model_path ,output_dir, phase='test'):
    # Parameters
    #def __init__(self, data_dir, image_size=(256, 256), batch_size=32, n_classes=2, n_channels=3,
    #              shuffle=True, level='L0', transform=None):
    valid_params = [
            os.path.join('..','data','extracted_patches',phase),
            (768,768),
            1,
            3,
            3,
            True,
            ]
    data_generator = load_dataset(valid_params)
    model = load_model(model_path,n_class=3)
    # X = np.zeros((1,256,256,3))
    # y_pred = model.predict(X, batch_size=1, verbose=0, steps=None)

    for i,(X, y) in enumerate(data_generator):
        y_pred = model.predict(X, batch_size=1, verbose=0, steps=None)
        X = (X*128+128).astype('uint8')[0]
        y = (y*255).astype('uint8')[0]
        y_pred = (y_pred * 255).astype('uint8')[0]
        print (X.shape, y.shape, y_pred.shape)
        n_classes = 3
        for j in range(n_classes):
            imsave(X, y[:,:,j], y_pred[:,:,j],out=os.path.join(output_dir,f"out_{i}_class_{j}.png"))

def predicted_from_wsi(output_dir,phase='test'):

    model = load_model(model_path,n_class=3)
    wsi_root_path = os.path.join('..','data','raw-data',phase)
    wsi_sample_ids = os.listdir(wsi_root_path)
    total_samples = len(wsi_sample_ids)
    for i,sample_id in enumerate(wsi_sample_ids):
        print(f'{i+1}/{total_samples} Running model on sample {sample_id}')
        sample_dir_path = os.path.join(wsi_root_path, sample_id)
        im_path = glob.glob(os.path.join(sample_dir_path,'*.svs'))[0]
        whole_mask_path = glob.glob(os.path.join(sample_dir_path,'*whole*'))[0]
        viable_mask_path = glob.glob(os.path.join(sample_dir_path,'*viable*'))[0]

        #Load images
        wsi_obj = openslide.OpenSlide(im_path)
        scaled_level = 2
        image_data = np.transpose(np.array(wsi_obj.read_region((0, 0),
                           scaled_level,
                           wsi_obj.level_dimensions[scaled_level]).convert('RGB')),
                           axes=[1, 0, 2])
        dims = wsi_obj.level_dimensions[0]

        def load_tiff(im_path):
            mask_image = io.imread(im_path)
            mask_image_scaled = cv2.resize(mask_image,wsi_obj.level_dimensions[scaled_level]).T
            return mask_image.T, mask_image_scaled

        
        whole_mask, whole_mask_scaled = load_tiff(whole_mask_path)
        viable_mask, viable_mask_scaled = load_tiff(viable_mask_path)

        patch_size = 768
        x_steps = dims[0]//patch_size
        y_steps = dims[1]//patch_size

        pred_whole_mask = np.zeros(dims)
        pred_viable_mask = np.zeros(dims)
        pred_bg_mask = np.zeros(dims)

        print(f'Running segmentaion algorithm patchwise')
        print(f'x_steps:{x_steps} y_steps:{y_steps}')
        for j in range(x_steps):
            print(f' X iteration {j+1}/{x_steps}')
            for k in range(y_steps):
                xc = j*768
                yc = k*768
                slide_patch = np.array(wsi_obj.read_region((xc,yc),0, (patch_size, patch_size)).convert('RGB')).transpose(1,0,2)[None,...]
                y_pred = model.predict(slide_patch,batch_size=1,verbose=0) 
                y_pred = (y_pred*255).astype('uint8')[0]
                pred_whole_mask_patch = y_pred[:,:,1]
                pred_viable_mask_patch = y_pred[:,:,2]
                pred_bg_mask_patch = y_pred[:,:,1]

                pred_whole_mask[xc:xc+patch_size,yc:yc+patch_size] = pred_whole_mask_patch 
                pred_viable_mask[xc:xc+patch_size,yc:yc+patch_size] = pred_viable_mask_patch 
                pred_bg_mask[xc:xc+patch_size,yc:yc+patch_size] = pred_bg_mask_patch 

                whole_mask_patch = whole_mask[xc:xc+patch_size,yc:yc+patch_size]
                viable_mask_patch = viable_mask[xc:xc+patch_size,yc:yc+patch_size]

            imsave(np.squeeze(slide_patch),whole_mask_patch,pred_whole_mask_patch,viable_mask_patch,pred_viable_mask_patch,out=os.path.join(output_dir,f'{sample_id}_{xc}_{yc}.png'))
        print(f'Saving scaled versions of the image')
        pred_whole_mask_scaled = cv2.resize(pred_whole_mask,wsi_obj.level_dimensions[scaled_level]).T
        pred_viable_mask_scaled = cv2.resize(pred_viable_mask,wsi_obj.level_dimensions[scaled_level]).T
        pred_bg_mask_scaled = cv2.resize(pred_bg_mask,wsi_obj.level_dimensions[scaled_level]).T

        file_namer = lambda x: os.path.join(output_dir, x)
        io.imsave(file_namer(f'{sample_id}_whole_mask_scaled.png'),pred_whole_mask_scaled)
        io.imsave(file_namer(f'{sample_id}_viable_mask_scaled.png'),pred_viable_mask_scaled)
        io.imsave(file_namer(f'{sample_id}_bg_mask_scaled.png'),pred_bg_mask_scaled)
        slide_patch_scaled = np.array(wsi_obj.read_region((0,0),2, wsi_obj.level_dimensions[2]).convert('RGB')).transpose(1,0,2)
        io.imsave(file_namer(f'{sample_id}_scaled.png'),slide_patch_scaled)

        print(f'Saving full-scale versions of the image')
        io.imsave(file_namer(f'{sample_id}_whole_mask.tiff'),pred_whole_mask)
        io.imsave(file_namer(f'{sample_id}_viable_mask.tiff'),pred_viable_mask)


if __name__ == '__main__':
    # Model Prediction
    saved_models_path = os.path.join('..','results','saved_models')
    train_jobs = os.listdir(saved_models_path)

    if train_jobs != [] :
        train_job_id = train_jobs[-1]
    else:
        print(f"No models found")

    train_job_id = 'expt_1/fully_trained_best_model'
    model_path = os.path.join(saved_models_path,train_job_id)
    if not os.path.isdir(model_path):
        print(f'Job: {train_job_id} not found')
        exit

    print(f"Testing with model stored in {train_job_id}")
    output_dir = os.path.join('..','results','predicted_images',train_job_id)
    os.makedirs(output_dir,exist_ok=True)

    predict_from_patches(model_path, output_dir)

    wsi_output_dir=os.path.join(output_dir,'wsi_output')
    os.makedirs(wsi_output_dir, exist_ok=True)
    predicted_from_wsi(wsi_output_dir)

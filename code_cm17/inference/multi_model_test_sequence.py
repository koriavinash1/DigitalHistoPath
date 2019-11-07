import sys
import os
import argparse
import logging
import json
import time
import numpy as np
import openslide
import PIL
import cv2
import matplotlib.pyplot as plt
import math
import json
import logging
import time
import tensorflow as tf
import gzip
import timeit

from scipy import stats
from tensorflow.keras import backend as K
from skimage.transform import resize, rescale
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *
from dataloader.inference_data_loader import WSIStridedPatchDataset
from models.seg_models import get_inception_resnet_v2_unet_softmax, unet_densenet121
from models.deeplabv3p_original import Deeplabv3
from models.utils import do_crf
from collections import OrderedDict
np.random.seed(0)

# python3 multi_model_test_sequence.py ../configs/Inference_Config.json ../../saved_models/keras_models/DFCN_UNET_CM17_RANDOM_16_NCRF_BCE_DICE_fold_1/model.10-0.24.h5 ../../saved_models/keras_models/IncpResV2_UNET_CM17_RANDOM_16_NCRF_BCE_DICE_fold_0/model.10-0.28.h5 ../../saved_models/keras_models/DeeplabV3p_CM17_RANDOM_16_NCRF_BCE_DICE_fold_2/model.09-0.28.h5
parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('model_path_DFCN', default=None, metavar='MODEL_PATH', type=str,
                    help='Path to the saved model weights file of a Keras model')
parser.add_argument('model_path_IRFCN', default=None, metavar='MODEL_PATH', type=str,
                    help='Path to the saved model weights file of a Keras model')
parser.add_argument('model_path_DLv3p', default=None, metavar='MODEL_PATH', type=str,
                    help='Path to the saved model weights file of a Keras model')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=4, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--level', default=6, type=int, help='heatmap generation level,'
                    ' default 6')
parser.add_argument('--sampling_stride', default=int(256//64), type=int, help='Sampling pixels in tissue mask,'
                    ' default 4')
parser.add_argument('--radius', default=12, type=int, help='radius for nms,'
                    ' default 12 (6 used in Google paper at level 7,'
                    ' i.e. inference stride = 128)')
parser.add_argument('--roi_masking', default=True, type=int, help='Sample pixels from tissue mask region,'
                    ' default True, points are not sampled from glass region')


def forward_transform(data, flip, rotate):
    """
    Do inverse data augmentation
    """
    if flip == 'FLIP_LEFT_RIGHT':
        data = np.fliplr(data)

    if rotate == 'ROTATE_90':
        data = np.rot90(data, 1)

    if rotate == 'ROTATE_180':
        data = np.rot90(data, 2)

    if rotate == 'ROTATE_270':
        data = np.rot90(data, 3)

    return data


def inverse_transform(data, flip, rotate):
    """
    Do inverse data augmentation
    """
    if flip == 'FLIP_LEFT_RIGHT':
        data = np.fliplr(data)

    if rotate == 'ROTATE_90':
        data = np.rot90(data, 3)

    if rotate == 'ROTATE_180':
        data = np.rot90(data, 2)

    if rotate == 'ROTATE_270':
        data = np.rot90(data, 1)

    return data

def get_index(coord_ax, probs_map_shape_ax, grid_ax):
    """
    This function checks whether coordinates are within the WSI
    """
    # print (coord_ax, probs_map_shape_ax, grid_ax)
    _min = grid_ax//2
    _max = grid_ax//2

    ax_min = coord_ax - _min
    while ax_min < 0:
        _min -= 1
        ax_min += 1

    ax_max = coord_ax + _max
    while ax_max > probs_map_shape_ax:
        _max -= 1
        ax_max -= 1

    return _min, _max

def get_wsi_cases(args, train_mode, model_name, dataset_name, patient_range, group_range):
    '''
    Get WSI cases with paths in a dictionary
    '''
    wsi_dic = OrderedDict()
    level = args.level 
    sampling_stride = args.sampling_stride
    npy_base_path = '../../predictions/{}/{}/level_{}_{}/npy'.format(model_name, dataset_name, str(level), str(sampling_stride))
    csv_base_path = '../../predictions/{}/{}/level_{}_{}/csv'.format(model_name, dataset_name, str(level), str(sampling_stride))
    png_base_path = '../../predictions/{}/{}/level_{}_{}/png'.format(model_name, dataset_name, str(level), str(sampling_stride))
    xml_base_path = '../../predictions/{}/{}/level_{}_{}/xml'.format(model_name, dataset_name, str(level), str(sampling_stride))

    l=patient_range[0];u=patient_range[1]
    tissue_mask_base_path_v1 = '../../data/TM_L{}_v1'.format(level)
    tissue_mask_base_path_v2 = '../../data/TM_L{}_v2'.format(level)
    if not os.path.exists(tissue_mask_base_path_v1):
        os.makedirs(tissue_mask_base_path_v1)

    if not os.path.exists(tissue_mask_base_path_v2):
        os.makedirs(tissue_mask_base_path_v2)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if train_mode:
        # for training
        label_base_path = cfg['cm17_train_annotation_path']
    else:
        # for testing
        l=patient_range[0];u=patient_range[1]

    if not os.path.exists(npy_base_path):
        os.makedirs(npy_base_path)

    if not os.path.exists(csv_base_path):
        os.makedirs(csv_base_path)

    if not os.path.exists(png_base_path):
        os.makedirs(png_base_path)

    if not os.path.exists(xml_base_path):
        os.makedirs(xml_base_path)

    # Numpy paths to 3 models from 3 folds and 1 ensemble model prediction
    model1_npy_path = os.path.join(npy_base_path, 'model1')
    if not os.path.exists(model1_npy_path):
        os.mkdir(model1_npy_path)
    model2_npy_path = os.path.join(npy_base_path, 'model2')                
    if not os.path.exists(model2_npy_path):
        os.mkdir(model2_npy_path)
    model3_npy_path = os.path.join(npy_base_path, 'model3')                
    if not os.path.exists(model3_npy_path):
        os.mkdir(model3_npy_path)
    ensemble_model_npy_path = os.path.join(npy_base_path, 'ensemble')                
    if not os.path.exists(ensemble_model_npy_path):
        os.mkdir(ensemble_model_npy_path)

    # Ensembled CRF labelled multiplies to prob_map at threshold 0.5 
    crf_model_npy_path = os.path.join(npy_base_path, 'ensemble_crf_l50')                
    if not os.path.exists(crf_model_npy_path):
        os.mkdir(crf_model_npy_path)

    for i in range(l,u):
        for j in range(group_range[0], group_range[1]):
            wsi_name = 'patient_{:03d}_node_{}'.format(i,j)
            path_dic = {}
            if train_mode:
                folder = 'center_'+str(int(i//20))
                wsi_path = cfg['cm17_train_data_path']+'/{}/patient_{:03d}_node_{}.tif'.format(folder,i,j)
                label_path = label_base_path + '/patient_{:03d}_node_{}_mask.tif'.format(i,j)
                if not os.path.exists(label_path):
                    label_path = None
            else:
                wsi_path = cfg['cm17_test_data_path']+'/patient_{:03d}_node_{}.tif'.format(i,j)
                label_path = None
            # Tissue Mask Generation
            mask_path_v1 = tissue_mask_base_path_v1+'/patient_{:03d}_node_{}.npy'.format(i,j)
            if not os.path.exists(mask_path_v1):
                slide = openslide.OpenSlide(wsi_path)
                tissue_mask_v1 = TissueMaskGeneration(slide, level)
                np.save(mask_path_v1, tissue_mask_v1)
                plt.imshow(tissue_mask_v1.T)
                plt.savefig(tissue_mask_base_path_v1 + '/' + os.path.basename(mask_path_v1).split('.')[0]+'.png')

            mask_path_v2 = tissue_mask_base_path_v2+'/patient_{:03d}_node_{}.npy'.format(i,j)
            if not os.path.exists(mask_path_v2):
                tissue_mask_v2 = TissueMaskGeneration_BIN_OTSU(slide, level)                
                mask_path_v2 = tissue_mask_base_path_v2+'/patient_{:03d}_node_{}.npy'.format(i,j)
                np.save(mask_path_v2, tissue_mask_v2)
                plt.imshow(tissue_mask_v2.T)            
                plt.savefig(tissue_mask_base_path_v2 + '/' + os.path.basename(mask_path_v2).split('.')[0]+'.png')

            # Save_path lists
            path_dic['wsi_path'] = wsi_path
            path_dic['label_path'] = label_path            
            path_dic['tissue_mask_path_v1'] = mask_path_v1
            path_dic['tissue_mask_path_v2'] = mask_path_v2
            path_dic['model1_path'] = model1_npy_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
            path_dic['model2_path'] = model2_npy_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
            path_dic['model3_path'] = model3_npy_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
            path_dic['ensemble_model_path'] = ensemble_model_npy_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
            path_dic['crf_model_path'] = crf_model_npy_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
            path_dic['png_ensemble_path'] = png_base_path + '/patient_{:03d}_node_{}_ensemble.png'.format(i,j)
            path_dic['png_ensemble_crf_path'] = png_base_path + '/patient_{:03d}_node_{}_ensemble_crf.png'.format(i,j)
            path_dic['csv_ensemble_path'] = csv_base_path + '/patient_{:03d}_node_{}.csv'.format(i,j)
            path_dic['xml_ensemble_path'] = xml_base_path + '/patient_{:03d}_node_{}.xml'.format(i,j)
            path_dic['csv_ensemble_crf_path'] = csv_base_path + '/patient_{:03d}_node_{}_crf.csv'.format(i,j)
            path_dic['xml_ensemble_crf_path'] = xml_base_path + '/patient_{:03d}_node_{}_crf.xml'.format(i,j)
            wsi_dic[wsi_name] = path_dic

    return wsi_dic

def rescale_image_intensity(image, factor=128):
    return np.uint8(image*128+128)

def get_probs_map(model_dic, dataloader, count_map_enabled=True):
    """
    Generate probability map
    """
    n_models = len(model_dic)
    probs_map = np.zeros((n_models,) + dataloader.dataset._mask.shape)
    label_map_t50 = np.zeros((n_models,) + dataloader.dataset._mask.shape, dtype=np.uint8)
    count_map = np.zeros((n_models,) + dataloader.dataset._mask.shape, dtype='uint8')
    num_batch = len(dataloader)
    batch_size = dataloader.batch_size
    map_x_size = dataloader.dataset._mask.shape[0]
    map_y_size = dataloader.dataset._mask.shape[1]
    level = dataloader.dataset._level
    # factor = dataloader.dataset._sampling_stride
    factor =  dataloader.dataset._image_size//pow(2, level)
    down_scale = 1.0 / pow(2, level)
    count = 0
    time_now = time.time()

    for (image_patches, x_coords, y_coords, label_patches) in dataloader:
        image_patches = image_patches.cpu().data.numpy()
        label_patches = label_patches.cpu().data.numpy()
        x_coords = x_coords.cpu().data.numpy()
        y_coords = y_coords.cpu().data.numpy()
        batch_size = image_patches.shape[0]
        for j in range(len(model_dic)):
            y_preds = model_dic[j].predict(image_patches, batch_size=batch_size, verbose=1, steps=None)         
            for i in range(batch_size):
                y_preds_rescaled = rescale(y_preds[i], down_scale, anti_aliasing=False)
                xmin, xmax = get_index(x_coords[i], map_x_size, factor)
                ymin, ymax = get_index(y_coords[i], map_y_size, factor)
                probs_map[j, x_coords[i] - xmin: x_coords[i] + xmax, y_coords[i] - ymin: y_coords[i] + ymax] +=\
                y_preds_rescaled[:,:,1].T[0:xmin+xmax, 0:ymin+ymax]
                count_map[j, x_coords[i] - xmin: x_coords[i] + xmax, y_coords[i] - ymin: y_coords[i] + ymax] +=\
                 np.ones_like(y_preds_rescaled[:,:,1].T[0:xmin+xmax, 0:ymin+ymax], dtype='uint8')
                label_t50 = labelthreshold(y_preds[i][:,:,1], threshold=.5)
                if np.sum(label_t50) >0:
                    MAP = do_crf(rescale_image_intensity(image_patches[i]), np.argmax(y_preds[i], axis=2), 2, enable_color=True, zero_unsure=False) 
                    MAP_rescaled = rescale(MAP, down_scale, order=0, preserve_range=True)
                else:
                    MAP_rescaled = np.zeros_like(y_preds_rescaled[:,:,1])
                label_map_t50[j, x_coords[i] - xmin: x_coords[i] + xmax, y_coords[i] - ymin: y_coords[i] + ymax] =\
                MAP_rescaled.T[0:xmin+xmax, 0:ymin+ymax]
        count += 1
        time_spent = time.time() - time_now
        time_now = time.time()
        print ('{}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), count, num_batch, time_spent))
    # imshow(count_map[0].T, count_map[1].T, count_map[2].T)        
    np.place(count_map, count_map==0, 1)
    probs_map /= count_map
    # imshow(dataloader.dataset._gt.T, probs_map[0].T, probs_map[1].T, probs_map[2].T, np.mean(probs_map, axis=0).T)

    return probs_map, label_map_t50

def make_dataloader(wsi_path, mask_path, label_path, args, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size']
    dataloader = DataLoader(WSIStridedPatchDataset(wsi_path, mask_path,
                            label_path,
                            image_size=cfg['image_size'],
                            normalize=True, flip=flip, rotate=rotate,
                            level=args.level, sampling_stride=args.sampling_stride, roi_masking=args.roi_masking),
                            batch_size=batch_size, num_workers=args.num_workers, drop_last=False)
    return dataloader

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True 
    session =tf.Session(config=core_config) 
    K.set_session(session)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    model_dic = {}
    batch_size = cfg['batch_size']
    image_size = cfg['image_size']

    if args.model_path_DFCN is not None:
        model = unet_densenet121((image_size, image_size), weights=None)
        model.load_weights(args.model_path_DFCN)
        print ("Loaded Model Weights from", args.model_path_DFCN)
        model_dic[0] = model
    if args.model_path_IRFCN is not None:
        model = get_inception_resnet_v2_unet_softmax((image_size, image_size), weights=None)
        model.load_weights(args.model_path_IRFCN)
        print ("Loaded Model Weights from", args.model_path_IRFCN)
        model_dic[1] = model
    if args.model_path_DLv3p is not None:
        model = Deeplabv3(input_shape=(image_size, image_size, 3), weights=None,\
                          classes=2,  activation = 'softmax', backbone='xception', OS=16)
        model.load_weights(args.model_path_DLv3p)
        print ("Loaded Model Weights from", args.model_path_DLv3p)
        model_dic[2] = model

    wsi_dic = get_wsi_cases(args, train_mode=False, model_name='Ensemble', dataset_name='CM17_Train', patient_range=(100,125), group_range=(0,5))

    for key in wsi_dic.keys():
        print ('Working on:', key)
        wsi_path = wsi_dic[key]['wsi_path']
        label_path = wsi_dic[key]['label_path']
        mask_path = wsi_dic[key]['tissue_mask_path_v2']

        if not os.path.exists(wsi_dic[key]['ensemble_model_path']):
            dataloader = make_dataloader(wsi_path, mask_path, label_path, args, cfg, flip='NONE', rotate='NONE')
            probs_map, label_t50_map = get_probs_map(model_dic, dataloader)

            # Saving the results
            np.save(wsi_dic[key]['model1_path'], probs_map[0])
            np.save(wsi_dic[key]['model2_path'], probs_map[1])
            np.save(wsi_dic[key]['model3_path'], probs_map[2])
            ensemble_prob_map = np.mean(probs_map, axis=0)
            np.save(wsi_dic[key]['ensemble_model_path'], ensemble_prob_map)
            voted_label_t50_map = np.sum(label_t50_map, axis=0)
            np.place(voted_label_t50_map, voted_label_t50_map==1,0) 
            np.place(voted_label_t50_map, voted_label_t50_map>1,1) 
            crf_ensemble_prob_map = ensemble_prob_map*voted_label_t50_map
            np.save(wsi_dic[key]['crf_model_path'], crf_ensemble_prob_map)

        if not os.path.exists(wsi_dic[key]['png_ensemble_path']):
            im = np.load(wsi_dic[key]['ensemble_model_path'])
            plt.imshow(im.T, cmap='jet')
            plt.savefig(wsi_dic[key]['png_ensemble_path'])
            im = np.load(wsi_dic[key]['crf_model_path'])
            plt.imshow(im.T, cmap='jet')
            plt.savefig(wsi_dic[key]['png_ensemble_crf_path'])

        if not os.path.exists(wsi_dic[key]['csv_ensemble_path']):
            nms_command = 'python3 nms.py'+' '+wsi_dic[key]['ensemble_model_path']+' '+wsi_dic[key]['csv_ensemble_path']+\
                        ' '+wsi_dic[key]['xml_ensemble_path']+' --level='+str(args.level)+' --radius='+str(args.radius)
            print (nms_command)
            os.system(nms_command)

        if not os.path.exists(wsi_dic[key]['csv_ensemble_crf_path']):
            nms_command = 'python3 nms.py'+' '+wsi_dic[key]['crf_model_path']+' '+wsi_dic[key]['csv_ensemble_crf_path']+\
                        ' '+wsi_dic[key]['xml_ensemble_crf_path']+' --level='+str(args.level)+' --radius='+str(args.radius)
            print (nms_command)
            os.system(nms_command)  

def main():
    t0 = timeit.default_timer()
    args = parser.parse_args()
    run(args)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))

if __name__ == '__main__':
    main()

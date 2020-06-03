#!/usr/bin/env python
import sys
import os
import argparse
import logging
import time
import glob
from shutil import copyfile
from multiprocessing import Pool, Value, Lock
import openslide
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import pandas as pd
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.measure import points_in_poly
from skimage import feature
from skimage.feature import canny
from sklearn.model_selection import KFold
import copy
import glob
import json
import random
import tqdm
from operator import itemgetter 
from collections import defaultdict
np.random.seed(0)
import math

parser = argparse.ArgumentParser()
parser.add_argument('mode' )
parser.add_argument('tumor_type')
args = parser.parse_args()
print(args)

n_samples = 50
train_n_patches = 2000
train_t_patches = 2000
total_train = train_n_patches+train_t_patches
print(total_train*n_samples)

valid_n_patches = train_n_patches//5
valid_t_patches = train_t_patches//5
total_valid = valid_n_patches+valid_t_patches
print(total_valid*n_samples)

data_path = os.path.join('..','..','data','raw-data','train')
out_path = os.path.join(data_path,'..','patch_coords_%dk'%(total_train*n_samples//1000))

if not os.path.isdir(out_path):
    os.makedirs(out_path)
ids = os.listdir(data_path)
mode = args.mode
tumor_type = args.tumor_type

# Functions
def ReadWholeSlideImage(image_path, level=None, RGB=True, read_image=True):
    """
    # =========================
    # Read Whole-Slide Image 
    # =========================
    """
    try:
        wsi_obj = openslide.OpenSlide(image_path)
        n_levels = wsi_obj.level_count
#         print("Number of Levels", n_levels)
#         print("Dimensions:%s, level_dimensions:%s"%(wsi_obj.dimensions, wsi_obj.level_dimensions))
#         print("Level_downsamples:", wsi_obj.level_downsamples)        
#         print("Properties", wsi_obj.properties)     
        if (level is None) or (level > n_levels-1):
            level = n_levels-1
#             print ('Default level selected', level)
        if read_image:
            if RGB:
                image_data = np.transpose(np.array(wsi_obj.read_region((0, 0),
                                   level,
                                   wsi_obj.level_dimensions[level]).convert('RGB')),
                                   axes=[1, 0, 2])
            else: 
                image_data = np.array(wsi_obj.read_region((0, 0),
                           level,
                           wsi_obj.level_dimensions[level]).convert('L')).T
        else:
            image_data = None 
#         print (image_data.shape)
    except openslide.OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None

    return wsi_obj, image_data, level

class Polygon(object):
    """
    Polygon represented as [N, 2] array of vertices
    """
    def __init__(self, name, vertices):
        """
        Initialize the polygon.

        Arguments:
            name: string, name of the polygon
            vertices: [N, 2] 2D numpy array of int
        """
        self._name = name
        self._vertices = vertices

    def __str__(self):
        return self._name

    def inside(self, coord):
        """
        Determine if a given coordinate is inside the polygon or not.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the polygon.
        """
        return points_in_poly([coord], self._vertices)[0]

    def vertices(self):

        return np.array(self._vertices)

class Annotation(object):
    """
    Annotation about the regions within BBOX in terms of vertices of polygons.
    """
    def __init__(self):
        self._bbox = []
        self._polygons_positive = []

    def __str__(self):
        return self._json_path

    def from_json(self, json_path):
        """
        Initialize the annotation from a json file.

        Arguments:
            json_path: string, path to the json annotation.
        """
        self._json_path = json_path
        with open(json_path) as f:
            annotations_json = json.load(f)

        for annotation in annotations_json['positive']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])      
            polygon = Polygon(name, vertices)
            if name == 'BBOX':
                self._bbox.append(polygon)
            else:
                self._polygons_positive.append(polygon)
                
    def inside_bbox(self, coord):
        """
        Determine if a given coordinate is inside the positive polygons of the annotation.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the positive/negative polygons of the
            annotation.
        """
        bboxes = copy.deepcopy(self._bbox)
        for bbox in bboxes:
            if bbox.inside(coord):
                return True
        return False
    
    def bbox_vertices(self):
        """
        Return the polygon represented as [N, 2] array of vertices

        Arguments:
            is_positive: bool, return positive or negative polygons.

        Returns:
            [N, 2] 2D array of int
        """
        return list(map(lambda x: x.vertices(), self._bbox))
    
    def inside_polygons(self, coord):
        """
        Determine if a given coordinate is inside the positive polygons of the annotation.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the positive/negative polygons of the
            annotation.
        """
        polygons = copy.deepcopy(self._polygons_positive)
        
        for polygon in polygons:
            if polygon.inside(coord):
                return True

        return False

    def polygon_vertices(self):
        """
        Return the polygon represented as [N, 2] array of vertices

        Arguments:
            is_positive: bool, return positive or negative polygons.

        Returns:
            [N, 2] 2D array of int
        """
        return list(map(lambda x: x.vertices(), self._polygons_positive))
    
def TissueMask(img_RGB, level):
    RGB_min = 50
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min
    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return tissue_mask

def ShuffleAndSampleFirstN(data, n=10):
    """
    Sampling by shuffling the data, then get only the first n elements.";
    """
    data=copy.deepcopy(data);
    random.shuffle(data);
    sample=data[0:n];
    return sample

def RandomUniformSample(data, n=1000, factor=1):
    data=copy.deepcopy(data);
    if len(data) <= n:
        sample_n = len(data)*factor        
    else:
        sample_n = n
        
    idxs = [];
    while len(idxs)<sample_n:
        rand=int(random.uniform(0, len(data)))
        if rand in idxs:
            pass
        else:
            idxs.append(rand);
    sample=[data[i] for i in idxs];
    return sample

def merge_files(file_list, output_file_path):
    with open(output_file_path, 'w') as outfile:
        for fname in file_list:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line) 
                    
def combine_text_files(files_dir_path, data_split_csv, output_file):
    """
    Combine all the files listed in data_split_csv from "files_dir_path" for CM17 dataset
    """
    files = []
    data_split_df = pd.read_csv(data_split_csv)
    for i in range(len(data_split_df.Image_Path)):
        file_path = os.path.join(files_dir_path, os.path.basename(data_split_df.Image_Path[i]).split('.')[0])
        files.append(file_path)
    mask_files = []
    for i in range(len(data_split_df.Mask_Path)):
        if data_split_df.Mask_Path[i] !='0':
            mask_dir = os.path.dirname(data_split_df.Mask_Path[i])
            mask_files.append(os.path.basename(data_split_df.Mask_Path[i]))
    image_dir = os.path.dirname(os.path.dirname(data_split_df.Image_Path[i]))            
    with open(output_file, 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                for line in infile:
                    pid, x_center, y_center = line.strip('\n').split(',')[0:3]
                    pid_no = int(pid.split('_')[1])
                    center_folder = 'center_'+str(int(pid_no//20))
                    pid_path = os.path.join(image_dir,center_folder,pid)
                    mask_name = pid.split('.')[0]+'_mask.tif'
                    if mask_name in mask_files:
                        mask_path = os.path.join(mask_dir, pid.split('.')[0]+'_mask.tif')
                        line = pid_path+','+mask_path+','+x_center+','+y_center+'\n'
                    else:
                        line = pid_path+','+str(0)+','+x_center+','+y_center+'\n'
                    outfile.write(line) 
                    
def threshold_img(img):
    '''
    Transforms a numpy array such that values greater than 0 are converted to 255
    '''
    img = np.array(img)
    np.place(img,img>0,255)
    return img
def extract_normal_patches_from_wsi(image_path, mask_path, json_path, out_path, mode, max_normal_points=1000):                    
    '''
    Extract Normal Patches coordinates and write to text file
    '''
    print('Extracting normal patches for %s' %(os.path.basename(image_path)))
    patch_level = 0
    patch_size = 256
    tumor_threshold = 0
    img_sampling_level = 2
    #Img downsamples are pows of 4, mask downsamples are pows of 2
    mask_sampling_level = int(math.sqrt(pow(4,img_sampling_level)))
    target_file = open(os.path.join(out_path, "{}_random_sample.txt".format(mode)), 'a')
    
    if os.path.exists(mask_path):
        print('True condition')
        wsi_obj, img_data, level = ReadWholeSlideImage(image_path, img_sampling_level, read_image=True)   
        mask_obj, mask_data, level = ReadWholeSlideImage(mask_path, mask_sampling_level)
#         if sampling_level > level:
#             sampling_level = level
        tissue_mask = TissueMask(img_data, img_sampling_level)
#         imshow(tissue_mask,threshold_img(mask_data))
        sampled_normal_pixels = np.transpose(np.nonzero(tissue_mask))
        
        # Perform Uniform sampling
        sampled_normal_pixels = RandomUniformSample(sampled_normal_pixels, 2*max_normal_points)
        sampled_normal_pixels_verified = []
        org_mag_factor = pow(4, img_sampling_level)                
        for coord in sampled_normal_pixels:   
            scoord = (int(coord[0]*org_mag_factor), int(coord[1]*org_mag_factor))
            shifted_point = (int(scoord[0]-patch_size//2), int(scoord[1]-patch_size//2))
            mask_patch = np.array(mask_obj.read_region(shifted_point, patch_level, (patch_size, patch_size)).convert('L'))        
            tumor_fraction = np.count_nonzero(mask_patch)/np.prod(mask_patch.shape) 
            if tumor_fraction <= tumor_threshold:
                sampled_normal_pixels_verified.append(scoord)
                slide_patch = np.array(wsi_obj.read_region(shifted_point, patch_level, (patch_size, patch_size)).convert('RGB'))
#                 imshow(slide_patch, mask_patch)
    else:
        print('False condition')
        mask_path = '0'        
        wsi_obj, img_data, level = ReadWholeSlideImage(image_path, sampling_level, read_image=True)   
        if sampling_level > level:
            sampling_level = level        
        tissue_mask = TissueMask(img_data, sampling_level)
#         imshow(tissue_mask)
        sampled_normal_pixels = list(np.transpose(np.nonzero(tissue_mask)))
        sampled_normal_pixels_verified = []
        org_mag_factor = pow(4, sampling_level)    
        for coord in sampled_normal_pixels:   
            scoord = (int(coord[0]*org_mag_factor), int(coord[1]*org_mag_factor))   
            sampled_normal_pixels_verified.append(scoord)
#         for coord in sampled_normal_pixels_verified:   
#             scaled_shifted_point = (int(coord[0]-patch_size//2), int(coord[1]-patch_size//2))
#             slide_patch = np.array(wsi_obj.read_region(scaled_shifted_point, patch_level, (patch_size, patch_size)).convert('RGB'))
#             imshow(slide_patch)
        
    # Perform Uniform sampling
    sampled_normal_pixels_verified = RandomUniformSample(sampled_normal_pixels_verified, max_normal_points)    
    for tpoint in sampled_normal_pixels_verified:
        target_file.write(image_path +','+mask_path +','+ str(tpoint[0]) + ',' + str(tpoint[1]))        
        target_file.write("\n")
    target_file.close()    
    no_samples = (len(sampled_normal_pixels_verified))                    
    print('Extracted %d normal samples' % (no_samples))
    return no_samples
                  
def extract_tumor_patches_from_wsi(image_path, mask_path, json_path, out_path, mode, max_tumor_points=2500):
    '''
    Extract Patches coordinates and write to text file
    '''
    print('Extracting tumor patches for %s' %(os.path.basename(image_path)))
    patch_size = 256
    patch_level = 0
    img_sampling_level = 2
    #Img downsamples are pows of 4, mask downsamples are pows of 2
    mask_sampling_level = int(math.sqrt(pow(4,img_sampling_level)))
    
    target_file = open(os.path.join(out_path, "{}_random_sample.txt".format(mode)), 'a')
    mask_obj, mask_data, level = ReadWholeSlideImage(mask_path, mask_sampling_level, RGB=False, read_image=True)
    org_mag_factor = pow(4, img_sampling_level)
    tumor_pixels = list(np.transpose(np.nonzero(mask_data)))
    tumor_pixels = RandomUniformSample(tumor_pixels, max_tumor_points) 
#     anno = Annotation()
#     anno.from_json(json_path)  
#     anno_vertices_list = list(anno.polygon_vertices())
#     anno_vertices_flat_list = [item for sublist in anno_vertices_list for item in sublist]
#     sampled_anno_vertices_flat_list = RandomUniformSample(anno_vertices_flat_list, max_tumor_points)        
    
    # Perform Uniform sampling    
    scaled_tumor_pixels = []
    for coord in list(tumor_pixels):    
        scoord = (int(coord[0]*org_mag_factor), int(coord[1]*org_mag_factor))   
        scaled_tumor_pixels.append(scoord)
                   
#     print ('Number of Tumor pixels', len(scaled_tumor_pixels))
#     scaled_tumor_pixels.extend(sampled_anno_vertices_flat_list)    
#     print ('Number of Tumor pixels+ vertices', len(scaled_tumor_pixels))
    
#     for coord in scaled_tumor_pixels:
#         print (coord)
#         scaled_shifted_point = (coord[0]-patch_size//2, coord[1]-patch_size//2)
#         wsi_obj, _, level = ReadWholeSlideImage(image_path, img_sampling_level, RGB=True, read_image=False)
#         slide_patch = np.array(wsi_obj.read_region(scaled_shifted_point, patch_level, (patch_size, patch_size)).convert('RGB'))
#         mask_patch = threshold_img(np.array(mask_obj.read_region(scaled_shifted_point, patch_level, (patch_size, patch_size)).convert('L')))
#         imshow(slide_patch, mask_patch)  
                
    for tpoint in scaled_tumor_pixels:
#         target_file.write(os.path.basename(image_path) +','+ str(tpoint[0]) + ',' + str(tpoint[1]))
        target_file.write(image_path +','+mask_path +','+ str(tpoint[0]) + ',' + str(tpoint[1]))        
        target_file.write("\n")

    target_file.close()
    no_samples = (len(scaled_tumor_pixels))
    print('Extracted %d tumor samples' % (no_samples))
    return no_samples

def batch_patch_gen(mode,tumor_type):
    count = 0
    if mode == 'train':
        n_patches = train_n_patches
        t_patches = train_t_patches
    elif mode == 'valid':
        n_patches = valid_n_patches
        t_patches = valid_t_patches
    else:
        return 0
    mode = '%s_paip_%s' % (mode,tumor_type)
    glob_str = '*%s*.tiff' % (tumor_type)
    for i,id in enumerate(ids):
        print('%d/%d : %s' %(i+1,len(ids),id))
        image_path = glob.glob(os.path.join(data_path,id,'*.svs'))[0]
        mask_path = glob.glob(os.path.join(data_path,id,glob_str))[0]
        abspath = os.path.abspath
        image_path = abspath(image_path)
        mask_path = abspath(mask_path)
        count+=extract_normal_patches_from_wsi(image_path, mask_path, None, out_path, mode,n_patches)
        if os.path.exists(mask_path):
            count+=extract_tumor_patches_from_wsi(image_path, mask_path, None, out_path, mode,t_patches)
    print ('Points sampled:', count)
    return '%s_paip_%s_random_sample.txt' % (mode,tumor_type)

def visualize(coord_file_path, patch_size=(256,256)):
    tumor_samples = 0
    fi = open(coord_file_path)
    for i, line in enumerate(fi):
        image_path, mask_path, x_center, y_center = line.strip('\n').split(',')[0:4]
        #print('%d %s'%(i,mask_path))
        x_top_left = int(int(x_center) - patch_size[0] / 2)
        y_top_left = int(int(y_center) - patch_size[1] / 2)            
        image_opslide = openslide.OpenSlide(image_path)
        image_data = image_opslide.read_region(
            (x_top_left, y_top_left), 0,
            patch_size).convert('RGB')        
        if mask_path != '0':                       
            x_top_left = int(int(x_center) - patch_size[0] / 2)
            y_top_left = int(int(y_center) - patch_size[1] / 2)            
            mask_obj = openslide.OpenSlide(mask_path)                                   
            mask_data = np.array(mask_obj.read_region((x_top_left, y_top_left),
                               0,
                               patch_size).convert('L'))
            np.place(mask_data,mask_data>0,255)
            fraction = np.count_nonzero(mask_data)/np.prod(mask_data.shape)
            if fraction > 0.0:
                imshow(image_data, mask_data)                   
        else:
            mask_data = np.zeros_like(image_data)
     
        if not i%1000:
            print(i)
            imshow(image_data, mask_data)
    fi.close()

def get_tumor_fraction(mask_image):
    fraction = np.count_nonzero(mask_image)/np.prod(mask_image.shape)
    return fraction
                                   
def add_tumor_fraction(coord_file_path, out_file_name, patch_size=(768,768)):
    tumor_samples = 0
    fi = open(coord_file_path)
    fo = open(os.path.dirname(coord_file_path)+'/'+ out_file_name, 'a')  
    for i,line in enumerate(fi):
        image_path, mask_path, x_center, y_center = line.strip('\n').split(',')[0:4]
        if mask_path != '0':                       
            x_top_left = int(int(x_center) - patch_size[0] / 2)
            y_top_left = int(int(y_center) - patch_size[1] / 2)            
            mask_obj = openslide.OpenSlide(mask_path)                                   
            mask_data = np.array(mask_obj.read_region((x_top_left, y_top_left),
                               0,
                               patch_size).convert('L'))                                   
            tumor_fraction = get_tumor_fraction(mask_data)
            if tumor_fraction > 0.0:
                tumor_samples += 1
#                 image_opslide = openslide.OpenSlide(image_path)
#                 image_data = image_opslide.read_region(
#                     (x_top_left, y_top_left), 0,
#                     patch_size).convert('RGB')
#                 print (mask_path, tumor_fraction)
#                 imshow(image_data, mask_data)
        else:
            tumor_fraction = 0
        fo.write(image_path +','+mask_path +','+x_center+','+y_center+','+str(tumor_fraction))        
        fo.write("\n")
    fo.close()
    fi.close()
    return tumor_samples

def wrapper_for_tumor_fraction(mode,tumor_type):
    train_coord_path = os.path.join(out_path,'%s_paip_%s_random_sample.txt' % (mode,tumor_type))
    train_tumor_count = add_tumor_fraction(train_coord_path, '%s_%s_tf.txt' % (mode,tumor_type))
    #print ('Train Stats:', 'Tumor_samples:', train_tumor_count, 'Normal_samples:', (train_ - train_tumor_count))
    
def split_df(df, column, save_dir, mode,threshold=0):
    df_tumor = df.loc[df[column]>threshold]
    df_normal = df.loc[df[column]==threshold]
    df_tumor.to_csv(os.path.join(save_dir,'{}_tumor.txt'.format(mode)), header=False, index=False)
    df_normal.to_csv(os.path.join(save_dir,'{}_normal.txt'.format(mode)), header=False, index=False)    
    return(df_tumor, df_normal)

def split_df_wrapper(mode,tumor_type):
    train_cm17_tf = os.path.join(out_path,'%s_%s_tf.txt' % (mode,tumor_type))
    train_cm17_tf_df = pd.read_csv(train_cm17_tf, names=['pid', 'mask', 'x', 'y', 'tf'])
    train_df_tumor, train_df_normal = split_df(train_cm17_tf_df, 'tf', out_path, '%s_%s'%(mode,tumor_type))
    print (len(train_df_tumor), len(train_df_normal))

    
print(f'Running patch extraction')
batch_patch_gen(mode,tumor_type)
print('Running tumor fraction calc')
wrapper_for_tumor_fraction(mode,tumor_type)
print('Running splitter')
split_df_wrapper(mode,tumor_type)

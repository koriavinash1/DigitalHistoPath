import os, sys
import numpy as np
import pandas as pd
import csv

import glob
import random
import cv2
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import scipy.stats.stats as st
from skimage.measure import label
from skimage.measure import regionprops

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *

N_FEATURES = 31
MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4

heatmap_feature_names = ['patient_name',
                         'region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor',
                         'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                         'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                         'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                         'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                         'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                         'solidity_skew', 'solidity_kurt', 'stage']


def get_image_open(wsi_path, level=None):
    try:
        wsi_image = OpenSlide(wsi_path)
        if level is None:
            level_used = wsi_image.level_count - 1
        else:
            level_used = level
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                   wsi_image.level_dimensions[level_used]))
        wsi_image.close()
    except OpenSlideUnsupportedFormatError:
        raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return image_open

def image_open(mask):
    mask = np.uint8(mask)
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    return image_open

def format_2f(number):
    return float("{0:.2f}".format(number))

def get_region_props(heatmap_threshold_2d, heatmap_prob_2d):
    labeled_img = label(heatmap_threshold_2d)
    return regionprops(labeled_img, intensity_image=heatmap_prob_2d)


def draw_bbox(heatmap_threshold, region_props, threshold_label='t90'):
    n_regions = len(region_props)
    print('No of regions(%s): %d' % (threshold_label, n_regions))
    for index in range(n_regions):
        print('\n\nDisplaying region: %d' % index)
        region = region_props[index]
        print('area: ', region['area'])
        print('bbox: ', region['bbox'])
        print('centroid: ', region['centroid'])
        print('convex_area: ', region['convex_area'])
        print('eccentricity: ', region['eccentricity'])
        print('extent: ', region['extent'])
        print('major_axis_length: ', region['major_axis_length'])
        print('minor_axis_length: ', region['minor_axis_length'])
        print('orientation: ', region['orientation'])
        print('perimeter: ', region['perimeter'])
        print('solidity: ', region['solidity'])

        cv2.rectangle(heatmap_threshold, (region['bbox'][1], region['bbox'][0]),
                      (region['bbox'][3], region['bbox'][2]), color=(0, 255, 0),
                      thickness=1)
        cv2.ellipse(heatmap_threshold, (int(region['centroid'][1]), int(region['centroid'][0])),
                    (int(region['major_axis_length'] / 2), int(region['minor_axis_length'] / 2)),
                    region['orientation'] * 90, 0, 360, color=(0, 0, 255),
                    thickness=2)

    cv2.imshow('bbox_%s' % threshold_label, heatmap_threshold)


def get_largest_tumor_index(region_props):
    largest_tumor_index = -1

    largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index

    return largest_tumor_index


def get_longest_axis_in_largest_tumor_region(region_props, largest_tumor_region_index):
    largest_tumor_region = region_props[largest_tumor_region_index]
    return max(largest_tumor_region['major_axis_length'], largest_tumor_region['minor_axis_length'])


def get_tumor_region_to_tissue_ratio(region_props, image_open):
    tissue_area = cv2.countNonZero(image_open)
    tumor_area = 0

    n_regions = len(region_props)
    for index in range(n_regions):
        tumor_area += region_props[index]['area']

    return float(tumor_area) / tissue_area


def get_tumor_region_to_bbox_ratio(region_props):
    # for all regions or largest region
    print()


def get_feature(region_props, n_region, feature_name):
    feature = [0] * 5
    if n_region > 0:
        feature_values = [region[feature_name] for region in region_props]
        feature[MAX] = format_2f(np.max(feature_values))
        feature[MEAN] = format_2f(np.mean(feature_values))
        feature[VARIANCE] = format_2f(np.var(feature_values))
        feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
        feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))

    return feature


def get_average_prediction_across_tumor_regions(region_props):
    # close 255
    region_mean_intensity = [region.mean_intensity for region in region_props]
    return np.mean(region_mean_intensity)


def extract_features(heatmap_prob, image_open):
    """
        Feature list:
        -> (01) given t = 0.90, total number of tumor regions
        -> (02) given t = 0.90, percentage of tumor region over the whole tissue region
        -> (03) given t = 0.50, the area of largest tumor region
        -> (04) given t = 0.50, the longest axis in the largest tumor region
        -> (05) given t = 0.90, total number pixels with probability greater than 0.90
        -> (06) given t = 0.90, average prediction across tumor region
        -> (07-11) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'area'
        -> (12-16) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'perimeter'
        -> (17-21) given t = 0.90, max, mean, variance, skewness, and kurtosis of  'compactness(eccentricity[?])'
        -> (22-26) given t = 0.50, max, mean, variance, skewness, and kurtosis of  'rectangularity(extent)'
        -> (27-31) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'solidity'
    :param heatmap_prob:
    :param image_open:
    :return:
    """

    heatmap_threshold_t90 = np.array(heatmap_prob)
    heatmap_threshold_t50 = np.array(heatmap_prob)
    heatmap_threshold_t90[heatmap_threshold_t90 < 0.90] = 0
    heatmap_threshold_t90[heatmap_threshold_t90 >= 0.90] = 255
    heatmap_threshold_t50[heatmap_threshold_t50 <= 0.50] = 0
    heatmap_threshold_t50[heatmap_threshold_t50 > 0.50] = 255

    heatmap_threshold_t90_2d = np.reshape(heatmap_threshold_t90,
                                          (heatmap_threshold_t90.shape[0], heatmap_threshold_t90.shape[1]))
    heatmap_threshold_t50_2d = np.reshape(heatmap_threshold_t50,
                                          (heatmap_threshold_t50.shape[0], heatmap_threshold_t50.shape[1]))
    heatmap_prob_2d = np.reshape(heatmap_prob,
                                 (heatmap_prob.shape[0], heatmap_prob.shape[1]))

    region_props_t90 = get_region_props(np.array(heatmap_threshold_t90_2d), heatmap_prob_2d)
    region_props_t50 = get_region_props(np.array(heatmap_threshold_t50_2d), heatmap_prob_2d)

    features = []

    f_count_tumor_region = len(region_props_t90)
    if f_count_tumor_region == 0:
        return [0.00] * N_FEATURES

    features.append(format_2f(f_count_tumor_region))

    f_percentage_tumor_over_tissue_region = get_tumor_region_to_tissue_ratio(region_props_t90, image_open)
    features.append(format_2f(f_percentage_tumor_over_tissue_region))

    largest_tumor_region_index_t90 = get_largest_tumor_index(region_props_t90)
    largest_tumor_region_index_t50 = get_largest_tumor_index(region_props_t50)
    f_area_largest_tumor_region_t50 = region_props_t50[largest_tumor_region_index_t50].area
    features.append(format_2f(f_area_largest_tumor_region_t50))

    f_longest_axis_largest_tumor_region_t50 = get_longest_axis_in_largest_tumor_region(region_props_t50,
                                                                                       largest_tumor_region_index_t50)
    features.append(format_2f(f_longest_axis_largest_tumor_region_t50))

    f_pixels_count_prob_gt_90 = cv2.countNonZero(heatmap_threshold_t90_2d)
    features.append(format_2f(f_pixels_count_prob_gt_90))

    f_avg_prediction_across_tumor_regions = get_average_prediction_across_tumor_regions(region_props_t90)
    features.append(format_2f(f_avg_prediction_across_tumor_regions))

    f_area = get_feature(region_props_t90, f_count_tumor_region, 'area')
    features += f_area

    f_perimeter = get_feature(region_props_t90, f_count_tumor_region, 'perimeter')
    features += f_perimeter

    f_eccentricity = get_feature(region_props_t90, f_count_tumor_region, 'eccentricity')
    features += f_eccentricity

    f_extent_t50 = get_feature(region_props_t50, len(region_props_t50), 'extent')
    features += f_extent_t50

    f_solidity = get_feature(region_props_t90, f_count_tumor_region, 'solidity')
    features += f_solidity

    # 
    # f_longest_axis_largest_tumor_region_t90 = get_longest_axis_in_largest_tumor_region(region_props_t90,
    #                                                                                    largest_tumor_region_index_t90)
    # f_area_larget_tumor_region_t90 = region_props_t90[largest_tumor_region_index_t90].area

    # cv2.imshow('heatmap_threshold_t90', heatmap_threshold_t90)
    # cv2.imshow('heatmap_threshold_t50', heatmap_threshold_t50)
    # draw_bbox(np.array(heatmap_threshold_t90), region_props_t90, threshold_label='t90')
    # draw_bbox(np.array(heatmap_threshold_t50), region_props_t50, threshold_label='t50')
    # key = cv2.waitKey(0) & 0xFF
    # if key == 27:  # escape
    #     exit(0)

    return features


if __name__ == '__main__':
    TRAIN = False
    # MODEL_NAME = 'DFCN_121_UNET' 
    MODEL_NAME = 'NCRF_CM16' 

    if TRAIN:
        stage_labels_path = '/media/mak/mirlproject1/CAMELYON17/training/groundtruth/stage_labels.csv'
        # heat_maps_path = '/media/mak/Data/Projects/Camelyon17/predictions/DenseNet-121_UNET/CM17_train/level_5_16/npy'
        heat_maps_path = '/media/mak/Data/Projects/Camelyon17/predictions/NCRF_CM17/training/resnet18_crf_inhouse_768_3_3/LEVEL_6_STRIDE_1/npy'
        tissue_map_level_5_path = '/media/mak/mirlproject1/CAMELYON17/training/dataset/TissueMask_Level_5'
        heat_maps_list = sorted(os.listdir(heat_maps_path))
        df_labels = pd.read_csv(stage_labels_path)

        f_train = '../predictions/CM17_TrainWSI_Features_{}.csv'.format(MODEL_NAME)
        features_file_train_all = open(f_train, 'w')
        wr_train = csv.writer(features_file_train_all, quoting=csv.QUOTE_NONNUMERIC)
        wr_train.writerow(heatmap_feature_names)

        for heat_map_file in heat_maps_list:
            print (heat_map_file)
            patient_node_name = heat_map_file.split('.')[0]
            patient_stage = df_labels['stage'][df_labels.patient[df_labels.patient == patient_node_name+'.tif'].index].tolist()

            heat_map = np.load(os.path.join(heat_maps_path, heat_map_file))
            tissue_map = np.load(os.path.join(tissue_map_level_5_path, heat_map_file))
            tissue_map = image_open(tissue_map)
            # imshow(heat_map.T, tissue_map.T)
            features = [patient_node_name]
            features += extract_features(heat_map, tissue_map)
            features += patient_stage
            wr_train.writerow(features)

    else:
        # heat_maps_path = '/media/mak/Data/Projects/Camelyon17/predictions/DenseNet-121_UNET/CM17_test/level_5_16/npy'
        heat_maps_path = '/media/mak/Data/Projects/Camelyon17/predictions/NCRF_CM17/testing/LEVEL_6_STRIDE_1/npy'
        tissue_map_level_5_path = '/media/mak/mirlproject1/CAMELYON17/testing/centers/TissueMask_Level_5'
        heat_maps_list = sorted(os.listdir(heat_maps_path))

        f_test = '../predictions/CM17_TestWSI_Features_{}.csv'.format(MODEL_NAME)
        features_file_test_all = open(f_test, 'w')
        wr_train = csv.writer(features_file_test_all, quoting=csv.QUOTE_NONNUMERIC)
        wr_train.writerow(heatmap_feature_names)

        for heat_map_file in heat_maps_list:
            print (heat_map_file)
            patient_node_name = heat_map_file.split('.')[0]
            patient_stage = 'Unknow'

            heat_map = np.load(os.path.join(heat_maps_path, heat_map_file))
            tissue_map = np.load(os.path.join(tissue_map_level_5_path, heat_map_file))
            tissue_map = image_open(tissue_map)
            # imshow(heat_map.T, tissue_map.T)
            features = [patient_node_name]
            features += extract_features(heat_map, tissue_map)
            features += patient_stage
            wr_train.writerow(features)



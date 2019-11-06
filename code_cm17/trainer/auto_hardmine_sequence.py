import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import xml.etree.cElementTree as ET
import pandas as pd
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

if __name__ == '__main__':
#    try:
        print ('Hardmining')
        model_name = 'DenseNet-121_UNET_CM16_NCRF'
        dataset_name = 'Hardmine_CM17'
        level = 5
        sampling_stride = 16
        csv_base_path = '../../datasetgen/{}/{}/level_{}_{}/csv'.format(model_name, dataset_name, str(level), str(sampling_stride))
        tissue_mask_base_path = '/media/mak/mirlproject1/CAMELYON17/training/dataset/TissueMask_Level_5'
        model_path = '/media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/CM16/unet_densenet121_imagenet_pretrained_L0_20190712-173828/Model_Stage2.h5'
        config_path = '../configs/DenseNet121_UNET_NCRF_CM16_COORDS_CDL_AUTOMINE.json'

        if not os.path.exists(csv_base_path):
            os.makedirs(csv_base_path)

        df = pd.read_csv('./annotated_train_data.csv')
        print(len(df['Image_Path']))
        for i in range (len(df['Image_Path'])):
            print(i)
            image_path = df['Image_Path'][i]
            label_path = df['Mask_Path'][i]
            if label_path == 'empty': 
                label_path = None

            csv_name = os.path.basename(image_path).split('.')[0]
            csv_path = os.path.join(csv_base_path, csv_name)
            mask_path = tissue_mask_base_path+'/{}.npy'.format(csv_name)

            command0 = '../inference/python3 tissue_mask_cm17.py'+' '+image_path+' '+mask_path+' '+'--level='+str(level) 
            if label_path is not None:
                command1 = 'python3 auto_hardmine.py'+' '+image_path+' '+model_path+' '+config_path+' '+csv_path \
                            +' '+'--mask_path='+mask_path+' --level='+str(level)+' --sampling_stride='+str(sampling_stride) \
                            +' '+'--label_path='+label_path
            else:
                command1 = 'python3 auto_hardmine.py'+' '+image_path+' '+model_path+' '+config_path+' '+csv_path \
                            +' '+'--mask_path='+mask_path+' --level='+str(level)+' --sampling_stride='+str(sampling_stride)

            if not os.path.exists(mask_path):
                print (command0)
                os.system(command0)
            if not os.path.exists(csv_path):
                print (command1)
                os.system(command1)
 #   except:
 #       print('Exception occured')
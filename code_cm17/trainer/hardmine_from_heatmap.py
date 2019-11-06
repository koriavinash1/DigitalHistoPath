import numpy as np 
import os, sys
import pandas as pd
import openslide
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from helpers.utils import *

PATCH_SIZE = 768
THRESHOLD = 0.4
LEVEL = 5

if __name__ == '__main__':

    heatmaps_path = '/media/mak/Data/Projects/Camelyon17/predictions/DenseNet-121_UNET/CM17_train/level_5_16/npy'
    out_csv_path = '/media/mak/Data/Projects/Camelyon17/code/keras_framework/datasetgen/DenseNet-121_UNET_CM16_NCRF/HARDMINE_CM17_HMAP_BASED/csv'

    if not os.path.exists(out_csv_path):
        os.makedirs(out_csv_path)

    df = pd.read_csv('./annotated_train_data.csv')
    for i in range (len(df['Image_Path'])):
        probs_map =  []
        patient_name = os.path.basename(df['Image_Path'][i]).split('.')[0]
        csv_path = os.path.join(out_csv_path, patient_name+'.txt')
        if not os.path.exists(csv_path):
            image_path = df['Image_Path'][i]
            label_path = df['Mask_Path'][i]
            if label_path == 'empty': 
                label_path = None
            print (patient_name)
            file_path = os.path.join(heatmaps_path, patient_name+'.npy')
            hmap = np.load(file_path)
            coords = np.where(hmap >= THRESHOLD)
            for j in range(len(coords[0])):
                x_coord = pow(2, LEVEL)*coords[0][j]
                y_coord = pow(2, LEVEL)*coords[1][j]
                # print (x_coord, y_coord)
                # img_obj = openslide.OpenSlide(image_path)
                # img = img_obj.read_region((x_coord - PATCH_SIZE//2, y_coord - PATCH_SIZE//2),
                # 0,
                # (PATCH_SIZE, PATCH_SIZE)).convert('RGB')

                if label_path is not None:
                    label_obj = openslide.OpenSlide(label_path)
                    label_img = label_obj.read_region((x_coord - PATCH_SIZE//2, y_coord - PATCH_SIZE//2),
                    0,
                    (PATCH_SIZE, PATCH_SIZE)).convert('L')
                else:
                    label_img = np.zeros((PATCH_SIZE, PATCH_SIZE))

                tumor_fraction = np.count_nonzero(label_img)/ np.prod(np.array(label_img).shape)
                if tumor_fraction < 0.05:
                    # imshow(img, label_img, title = ['Image', str(tumor_fraction)+ str()])
                    probs_map.append((patient_name, str(x_coord), str(y_coord), str(tumor_fraction)))

            with open(csv_path, 'w') as out:
                csv_out = csv.writer(out)
                for row in probs_map:
                    csv_out.writerow(row)  

    
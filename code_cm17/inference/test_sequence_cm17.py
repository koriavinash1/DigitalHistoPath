import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import xml.etree.cElementTree as ET
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')



if __name__ == '__main__':
    try:
        TRAIN = True
        model_name = 'DenseNet-121_UNET'
        dataset_name = 'CM17_train'
        level = 5
        sampling_stride = 16
        radius = 24
        npy_base_path = '../../../../predictions/{}/{}/level_{}_{}/npy'.format(model_name, dataset_name, str(level), str(sampling_stride))
        csv_base_path = '../../../../predictions/{}/{}/level_{}_{}/csv'.format(model_name, dataset_name, str(level), str(sampling_stride))
        png_base_path = '../../../../predictions/{}/{}/level_{}_{}/png'.format(model_name, dataset_name, str(level), str(sampling_stride))
        xml_base_path = '../../../../predictions/{}/{}/level_{}_{}/xml'.format(model_name, dataset_name, str(level), str(sampling_stride))

        if TRAIN:
            # for training
            tissue_mask_base_path = '/media/mak/mirlproject1/CAMELYON17/training/dataset/TissueMask_Level_5'
            label_base_path = '/media/mak/mirlproject1/CAMELYON17/training/groundtruth/lesion_annotations/Mask'
            l=0;u=100
        else:
            # for testing
            tissue_mask_base_path = '/media/mak/mirlproject1/CAMELYON17/testing/centers/TissueMask_Level_5'
            l=100;u=200

        if not os.path.exists(npy_base_path):
            os.makedirs(npy_base_path)

        if not os.path.exists(csv_base_path):
            os.makedirs(csv_base_path)

        if not os.path.exists(png_base_path):
            os.makedirs(png_base_path)

        if not os.path.exists(xml_base_path):
            os.makedirs(xml_base_path)

        if not os.path.exists(tissue_mask_base_path):
            os.makedirs(tissue_mask_base_path)

        for i in range(l,u):
            for j in range(5):
                if TRAIN:
                    folder = 'center_'+str(int(i//20))
                    image_path = '/media/mak/mirlproject1/CAMELYON17/training/dataset/{}/patient_{:03d}_node_{}.tif'.format(folder,i,j)
                else:
                    image_path = '/media/mak/mirlproject1/CAMELYON17/testing/centers/dataset/patient_{:03d}_node_{}.tif'.format(i,j)

                model_path = '/media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/CM16/unet_densenet121_imagenet_pretrained_L0_20190712-173828/Model_Stage2.h5'

                config_path = '../configs/DenseNet121_UNET_NCRF_CM16_COORDS_CDL.json'

                mask_path = tissue_mask_base_path+'/patient_{:03d}_node_{}.npy'.format(i,j)
                npy_path = npy_base_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
                csv_path = csv_base_path + '/patient_{:03d}_node_{}.csv'.format(i,j)
                png_path = png_base_path + '/patient_{:03d}_node_{}.png'.format(i,j)
                xml_path = xml_base_path + '/patient_{:03d}_node_{}.xml'.format(i,j)

                if TRAIN:
                    label_path = label_base_path + '/patient_{:03d}_node_{}.tif'.format(i,j)

                command0 = 'python3 tissue_mask_cm17.py'+' '+image_path+' '+mask_path+' '+'--level='+str(level) 
                if os.path.exists(label_path):
                    command1 = 'python3 probs_map.py'+' '+image_path+' '+model_path+' '+config_path+' '+npy_path \
                                +' '+'--mask_path='+mask_path+' --level='+str(level)+' --sampling_stride='+str(sampling_stride) \
                                +' '+'--label_path='+label_path
                else:
                    command1 = 'python3 probs_map.py'+' '+image_path+' '+model_path+' '+config_path+' '+npy_path \
                                +' '+'--mask_path='+mask_path+' --level='+str(level)+' --sampling_stride='+str(sampling_stride)

                command2 = 'python3 nms.py'+' '+npy_path+' '+csv_path+' '+xml_path+' --level='+str(level)+' --radius='+str(radius)

                if not os.path.exists(mask_path):
                    print (command0)
                    os.system(command0)
                if not os.path.exists(npy_path):
                    print (command1)
                    os.system(command1)
                if not os.path.exists(png_path):
                    im = np.load(npy_path)
                    plt.imshow(im.T, cmap='jet')
                    plt.savefig(png_path)
                if not os.path.exists(csv_path):
                    print (command2)
                    os.system(command2)


        TRAIN = False

        model_name = 'DenseNet-121_UNET'
        dataset_name = 'CM17_test'
        level = 5
        sampling_stride = 16
        radius = 24
        npy_base_path = '../../../../predictions/{}/{}/level_{}_{}/npy'.format(model_name, dataset_name, str(level), str(sampling_stride))
        csv_base_path = '../../../../predictions/{}/{}/level_{}_{}/csv'.format(model_name, dataset_name, str(level), str(sampling_stride))
        png_base_path = '../../../../predictions/{}/{}/level_{}_{}/png'.format(model_name, dataset_name, str(level), str(sampling_stride))
        xml_base_path = '../../../../predictions/{}/{}/level_{}_{}/xml'.format(model_name, dataset_name, str(level), str(sampling_stride))

        if TRAIN:
            # for training
            tissue_mask_base_path = '/media/mak/mirlproject1/CAMELYON17/training/dataset/TissueMask_Level_5'
            label_base_path = '/media/mak/mirlproject1/CAMELYON17/training/groundtruth/lesion_annotations/Mask'
            l=0;u=100
        else:
            # for testing
            tissue_mask_base_path = '/media/mak/mirlproject1/CAMELYON17/testing/centers/TissueMask_Level_5'
            l=100;u=200

        if not os.path.exists(npy_base_path):
            os.makedirs(npy_base_path)

        if not os.path.exists(csv_base_path):
            os.makedirs(csv_base_path)

        if not os.path.exists(png_base_path):
            os.makedirs(png_base_path)

        if not os.path.exists(xml_base_path):
            os.makedirs(xml_base_path)

        if not os.path.exists(tissue_mask_base_path):
            os.makedirs(tissue_mask_base_path)

        for i in range(l,u):
            for j in range(5):
                if TRAIN:
                    folder = 'center_'+str(int(i//20))
                    image_path = '/media/mak/mirlproject1/CAMELYON17/training/dataset/{}/patient_{:03d}_node_{}.tif'.format(folder,i,j)
                else:
                    image_path = '/media/mak/mirlproject1/CAMELYON17/testing/centers/dataset/patient_{:03d}_node_{}.tif'.format(i,j)

                model_path = '/media/mak/Data/Projects/Camelyon17/saved_models/keras_models/segmentation/CM16/unet_densenet121_imagenet_pretrained_L0_20190712-173828/Model_Stage2.h5'

                config_path = '../configs/DenseNet121_UNET_NCRF_CM16_COORDS_CDL.json'

                mask_path = tissue_mask_base_path+'/patient_{:03d}_node_{}.npy'.format(i,j)
                npy_path = npy_base_path + '/patient_{:03d}_node_{}.npy'.format(i,j)
                csv_path = csv_base_path + '/patient_{:03d}_node_{}.csv'.format(i,j)
                png_path = png_base_path + '/patient_{:03d}_node_{}.png'.format(i,j)
                xml_path = xml_base_path + '/patient_{:03d}_node_{}.xml'.format(i,j)

                if TRAIN:
                    label_path = label_base_path + '/patient_{:03d}_node_{}.tif'.format(i,j)

                command0 = 'python3 tissue_mask_cm17.py'+' '+image_path+' '+mask_path+' '+'--level='+str(level) 
                if os.path.exists(label_path):
                    command1 = 'python3 probs_map.py'+' '+image_path+' '+model_path+' '+config_path+' '+npy_path \
                                +' '+'--mask_path='+mask_path+' --level='+str(level)+' --sampling_stride='+str(sampling_stride) \
                                +' '+'--label_path='+label_path
                else:
                    command1 = 'python3 probs_map.py'+' '+image_path+' '+model_path+' '+config_path+' '+npy_path \
                                +' '+'--mask_path='+mask_path+' --level='+str(level)+' --sampling_stride='+str(sampling_stride)

                command2 = 'python3 nms.py'+' '+npy_path+' '+csv_path+' '+xml_path+' --level='+str(level)+' --radius='+str(radius)

                if not os.path.exists(mask_path):
                    print (command0)
                    os.system(command0)
                if not os.path.exists(npy_path):
                    print (command1)
                    os.system(command1)
                if not os.path.exists(png_path):
                    im = np.load(npy_path)
                    plt.imshow(im.T, cmap='jet')
                    plt.savefig(png_path)
                if not os.path.exists(csv_path):
                    print (command2)
                    os.system(command2)
    except:
        pass
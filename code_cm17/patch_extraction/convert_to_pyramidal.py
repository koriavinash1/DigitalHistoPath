import os, sys
import numpy as np
import json
import glob
import time

__all__ = ["create_pyramidal_img",
            "load_wsi_head",
            "load_wsi_level_img",
            ]

def create_pyramidal_img(img_path, save_dir):
    """ Convert normal image to pyramidal image.
    Parameters
    -------
    img_path: str
        Whole slide image path (absolute path is needed)
    save_dir: str
        Location of the saved the generated pyramidal image with extension tiff,
        (absolute path is needed)
    Returns
    -------
    status: int
        The status of the pyramidal image generation (0 stands for success)
    Notes
    -------
    ImageMagick need to be preinstalled to use this function.
    >>> sudo apt-get install imagemagick
    Examples
    --------
    >>> img_path = os.path.join(PRJ_PATH, "test/data/Images/CropBreastSlide.tif")
    >>> save_dir = os.path.join(PRJ_PATH, "test/data/Slides")
    >>> status = pyramid.create_pyramidal_img(img_path, save_dir)
    >>> assert status == 0
    """

    convert_cmd = "convert " + img_path
    convert_option = " -compress LZW -define tiff:tile-geometry=256x256 ptif:"
    img_name = os.path.basename(img_path)
    convert_dst = os.path.join(save_dir, os.path.splitext(img_name)[0] + ".tiff")
    status = os.system(convert_cmd + convert_option + convert_dst)

    return status

def save_to_json(json_dict):
    phase = 'train'
    lock_file = 'lock.json'
    with open(lock_file,'w') as json_file:
        json.dump(json_dict,json_file)

def read_json():
    phase = 'train'
    lock_file = 'lock.json'
    with open(lock_file,'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict

if __name__ == '__main__':
    phase = 'train'
    lock_file = 'lock.json'
    if os.path.isfile(lock_file):
        json_dict = read_json()
        list_of_tifs = glob.glob(os.path.join(phase,'**','*.tif'))
        list_of_remaining_tifs = list(set(list_of_tifs)-set(json_dict['completed'])-set(json_dict['inprog']))
        total = len(list_of_remaining_tifs)

        while len(list_of_remaining_tifs) > 0 :
            tif = list_of_remaining_tifs[0]
            json_dict['inprog'].append(tif)
            save_to_json(json_dict)

            print('%d left | doing %s' %(total,tif))
            stime = time.time()
            create_pyramidal_img(tif, os.path.dirname(tif))
            ftime = time.time()
            print('Completed in %.3f'% ((ftime-stime)/60))

            json_dict['completed'].append(tif)
            json_dict['inprog'].remove(tif)
            save_to_json(json_dict)

            json_dict = read_json()
            list_of_tifs = glob.glob(os.path.join(phase,'**','*.tif'))
            list_of_remaining_tifs = list(set(list_of_tifs)-set(json_dict['completed']))
            total = len(list_of_remaining_tifs)
    else:
        json_dict = {'completed':[]}
        save_to_json(json_dict)

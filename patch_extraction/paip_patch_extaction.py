"""
Creates random patches from sample image and mask images. The code is specific to the PAIP dataset of MICCAI.
"""
import sys
import os
import copy
import glob
import logging
import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage import io, transform
from PIL import Image

import random
import cv2
np.random.seed(0)
#logging.basicConfig(level=logging.INFO)

empty_masks = []
error_samples = []

# Functions
def ReadWholeSlideImage(image_path, level=None):
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
            print ('Default')
            level = n_levels-1
        image_data = np.transpose(np.array(wsi_obj.read_region((0, 0),
                           level,
                           wsi_obj.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])
        #WHY the transpose?
#         print (image_data.shape)
    except openslide.OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None

    return wsi_obj, image_data, level

def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_off = kwargs.get('axis_off','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
            if axis_off: 
              plt.axis('off')  
    if kwargs.get("saveto",'')=='':
        plt.show()
    else:
        plt.savefig(kwargs.get("saveto",''))
 
def imsave(*args, **kwargs):
    """
    Concatenate the images given in args and saves them as a single image in the specified output destination.
    Images should be numpy arrays and have same dimensions along the 0 axis.
    imsave(im1,im2,out="sample.png")
    """
    args = list(args)
    for i in range(len(args)):
        if type(args[i]) != np.ndarray:
            logging.error("Not a numpy array. Aborting.")
            return 0
        if len(args[i].shape) == 2:
            args[i] = np.dstack([args[i]]*3)
            if args[i].max() == 1:
               args[i] = args[i]*255
            
    out_destination = kwargs.get("out")
    try:
        concatenated_arr = np.concatenate(args,axis=1)
    except ValueError as e:
        
        print(e)
        return 0
    im = Image.fromarray(concatenated_arr)
    if not kwargs.get('silent', False):
        print(f"Saving to {out_destination}")
    im.save(out_destination)
    
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

def TissueMask(image_path, level):
    RGB_min = 50
    slide = openslide.OpenSlide(image_path)
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                           level,
                           slide.level_dimensions[level]).convert('RGB')),
                           axes=[0, 1, 2])
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
    return tissue_mask.T

def ShuffleAndSampleFirstN(data, n=10):
    """
    Sampling by shuffling the data, then get only the first n elements.";
    """
    data=copy.deepcopy(data);
    random.shuffle(data);
    sample=data[0:n];
    return sample

def RandomUniformSample(data, n=10, factor=1):
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

def extract_patches_from_wsi(image_path, whole_mask_path, viable_mask_path, out_path, out_prefix):
    '''
    Extract Patches coordinates and write to text file
    '''
    patch_size = 768
    patch_level = 0
    sampling_level = 2
    max_whole_tumor_points = 1000
    max_viable_tumor_points = 1000
    max_non_tumor_points = 1000
    tumor_threshold = 0.3
    center_crop_size = 64
    crop_fracl = (1/2*(1-center_crop_size/patch_size))
    crop_frach = (1/2*(1+center_crop_size/patch_size))

    #target_normal = open(os.path.join(out_path, "normal_{}.txt".format(mode)), 'a')
    #target_tumor = open(os.path.join(out_path, "tumor_{}.txt".format(mode)), 'a')
   
    logging.info("Reading image and mask")
    logging.info("Loading tumor image")
    try:
        wsi_obj, img_data, level = ReadWholeSlideImage(image_path, sampling_level)
    except Exception as e:
        print(f"Error ocurred: {e}")
        error_samples.append(out_prefix)
        return 0
    
    def load_tiff(im_path):
        mask_image = io.imread(im_path)
        mask_image_scaled = cv2.resize(mask_image,wsi_obj.level_dimensions[sampling_level]).T
        return mask_image.T, mask_image_scaled
    
    logging.info("Loading mask images")
    whole_mask, whole_mask_scaled = load_tiff(whole_mask_path)
    viable_mask, viable_mask_scaled = load_tiff(viable_mask_path)
    imsave(img_data,whole_mask_scaled,viable_mask_scaled,out=os.path.join(out_path,"reference.jpg"), silent=True)
    logging.info(f"wsi dimensions: {wsi_obj.level_dimensions}")
    logging.info(f"whole mask shapes: {whole_mask.shape} {whole_mask_scaled.shape}")
    logging.info(f"viable mask shapes: {viable_mask.shape} {viable_mask_scaled.shape}")
    logging.info(f"Scaled images max vals: {whole_mask_scaled.max()}, {viable_mask_scaled.max()}")
    logging.info(f"Full size images max vals: {whole_mask.max()}, {viable_mask.max()}")

    logging.info("Creating tissue mask")
    tissue_mask = TissueMask(image_path, sampling_level)
    whole_tumor_pixels = np.transpose(np.nonzero(whole_mask_scaled))
    viable_tumor_pixels = np.transpose(np.nonzero(viable_mask_scaled))
    tissue_pixels = np.transpose(np.nonzero(tissue_mask))
    
    logging.info("Performing unform random sample")
    sampled_whole_tumor_pixels = RandomUniformSample(whole_tumor_pixels, max_whole_tumor_points)    
    sampled_viable_tumor_pixels = RandomUniformSample(viable_tumor_pixels, max_viable_tumor_points)    
    sampled_tissue_pixels = RandomUniformSample(tissue_pixels, max_non_tumor_points)
    org_mag_factor = wsi_obj.level_downsamples[sampling_level]
    pixels_tosample = sampled_viable_tumor_pixels+sampled_whole_tumor_pixels
    total_points = len(pixels_tosample)

    if total_points == 0:
        print("Empty mask read")
        import ipdb;ipdb.set_trace(); 
        empty_masks.append(out_prefix)
    logging.info(f"Whole patches: {len(sampled_whole_tumor_pixels)} {len(sampled_viable_tumor_pixels)}")

    for i,point in enumerate(pixels_tosample):
        plt.close("all")
        logging.info(f"Finished patch {i+1}/{total_points}")
        print(f"Finished patch {i+1}/{total_points}", end='\r')
        #magnify_point = lambda x: int(x*org_mag_factor) - patch_size//2
        magnify_point = lambda x: int(x*org_mag_factor) 
        xc = magnify_point(point[0])
        yc = magnify_point(point[1])
        
        x_dim,y_dim = wsi_obj.level_dimensions[0]
        if xc+patch_size>x_dim:
            print(f"x overflew {xc,x_dim}")
            xc = x_dim - patch_size
        if yc+patch_size>y_dim:
            print(f"y overflew {yc,y_dim}")
            yc = y_dim - patch_size
        scaled_shifted_point = (xc,yc)
        
        slide_patch = np.array(wsi_obj.read_region(scaled_shifted_point, patch_level, (patch_size, patch_size)).convert('RGB')).transpose(1,0,2)
        whole_mask_patch = whole_mask[xc:xc+patch_size,yc:yc+patch_size]
        viable_mask_patch = viable_mask[xc:xc+patch_size,yc:yc+patch_size]
        #print(f"x,y {scaled_shifted_point} | {patch_size}")
        #imshow(slide_patch, whole_mask_patch, viable_mask_patch,saveto=os.path.join(out_path,f"{i}.jpg"))
        try:
            imsave(slide_patch, whole_mask_patch, viable_mask_patch,out=os.path.join(out_path,f"{out_prefix}_{i+1}.png"),silent=True)
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
    del whole_mask,viable_mask,whole_mask_scaled,viable_mask_scaled

data_dir = os.path.join('..','data')
path_prefix = os.path.join(data_dir,'raw-data')
sample_image_path = path_prefix+"01_01_0083.svs"
sample_whole_mask_path = path_prefix+"01_01_0083_whole.tif"
sample_viable_mask_path = path_prefix+"01_01_0083_viable.tif"
sample_whole_xml_path=path_prefix+"01_01_0083.xml"

mode="train"
json_path=""
out_path = os.path.join(data_dir,'extracted_patches')
sample_dirs = os.listdir(path_prefix)
total_samples = len(sample_dirs)
for i,sample_dir in enumerate(sample_dirs):
    print(f"{i+1}/{total_samples} Creating patches from sample {sample_dir}")
    sample_dir_path = os.path.join(path_prefix,sample_dir)
    try:
         im_path = glob.glob(os.path.join(sample_dir_path,'*.svs'))[0]
    except Exception as e:
         print(e)
         import ipdb; ipdb.set_trace()
    whole_mask_path = glob.glob(os.path.join(sample_dir_path,'*whole*'))[0]
    viable_mask_path = glob.glob(os.path.join(sample_dir_path,'*viable*'))[0]
    sample_out_path = os.path.join(out_path, sample_dir)
    if not os.path.exists(sample_out_path):
        os.makedirs(sample_out_path)
    extract_patches_from_wsi(im_path, whole_mask_path, viable_mask_path, sample_out_path, sample_dir)

print(f"Empty mask images list {empty_masks}")
print(f"Error images list {error_samples}")

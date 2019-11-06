import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import xml.etree.cElementTree as ET
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

def BinMorphoProcessMask(mask):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    return image_open

def get_bbox(cont_img, rgb_image=None):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (0, 0, 255)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    for x, y, h, w in bounding_boxes:
        rgb_contour = cv2.rectangle(rgb_contour,(x,y),(x+h,y+w),(0,255,0),2)
    return bounding_boxes, rgb_contour

def get_all_bbox_masks_with_stride(mask, stride_factor):
    """
    Find the bbox and corresponding masks
    """
    bbox_mask = np.zeros_like(mask)
    bounding_boxes, _ = get_bbox(mask)
    y_size, x_size = bbox_mask.shape
    for x, y, h, w in bounding_boxes:
        x_min = x - stride_factor
        x_max = x + h + stride_factor
        y_min = y - stride_factor
        y_max = y + w + stride_factor
        if x_min < 0: 
         x_min = 0
        if y_min < 0: 
         y_min = 0
        if x_max > x_size: 
         x_max = x_size - 1
        if y_max > y_size: 
         y_max = y_size - 1      
        bbox_mask[y_min:y_max:stride_factor, x_min:x_max:stride_factor]=1
        
    return bbox_mask


def get_all_bbox_masks(mask, stride_factor):
    """
    Find the bbox and corresponding masks
    """
    bbox_mask = np.zeros_like(mask)
    bounding_boxes, _ = get_bbox(mask)
    y_size, x_size = bbox_mask.shape
    for x, y, h, w in bounding_boxes:
        x_min = x - stride_factor
        x_max = x + h + stride_factor
        y_min = y - stride_factor
        y_max = y + w + stride_factor
        if x_min < 0: 
         x_min = 0
        if y_min < 0: 
         y_min = 0
        if x_max > x_size: 
         x_max = x_size - 1
        if y_max > y_size: 
         y_max = y_size - 1      
        bbox_mask[y_min:y_max, x_min:x_max]=1
    return bbox_mask

def find_largest_bbox(mask, stride_factor):
    """
    Find the largest bounding box encompassing all the blobs
    """
    y_size, x_size = mask.shape
    x, y = np.where(mask==1)
    bbox_mask = np.zeros_like(mask)
    x_min = np.min(x) - stride_factor
    x_max = np.max(x) + stride_factor
    y_min = np.min(y) - stride_factor
    y_max = np.max(y) + stride_factor
    
    if x_min < 0: 
     x_min = 0
    
    if y_min < 0: 
     y_min = 0

    if x_max > x_size: 
     x_max = x_size - 1
    
    if y_max > y_size: 
     y_max = y_size - 1    
    
    # print(x_min, x_max, y_min, y_max)
    bbox_mask[x_min:x_max, y_min:y_max]=1
    return bbox_mask

def TissueMaskGeneration(slide_obj, level, RGB_min=50):
    img_RGB = np.transpose(np.array(slide_obj.read_region((0, 0),
                       level,
                       slide_obj.level_dimensions[level]).convert('RGB')),
                       axes=[1, 0, 2])
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

def GenerateXMLfromCSV(csvpath, outxmlpath):
    """Reads the data inside CSV file and generates xml file for visualizing via ASAP
    
    Args:
        csvpath: 
        outxmlpath: 
        
    Returns:
        None
    """
    Annotations_root = ET.Element("ASAP_Annotations")
    Annotations = ET.SubElement(Annotations_root, "Annotations")

    csv_lines = open(csvpath,"r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Name_value = "Annotation {}".format(i)
        X_value = elems[1]
        Y_value = elems[2]

        Annotation =  ET.SubElement(Annotations, "Annotation", Name=Name_value, Type="Dot", PartOfGroup="None", Color="#F4FA58")
        Coordinates =  ET.SubElement(Annotation, "Coordinates")
        Coordinate =  ET.SubElement(Coordinates, "Coordinate", Order="0", X=X_value, Y=Y_value)

    AnnotationGroups = ET.SubElement(Annotations_root, "AnnotationGroups")
    tree = ET.ElementTree(Annotations_root)
    tree.write(outxmlpath)
    
def grid_to_single(image_batch, label_image=False):
    shape = image_batch.shape
    # print (shape)
    n = int(np.sqrt(shape[0]))
    x = shape[1]
    y = shape[2]
    if not label_image:
        img_array = np.zeros((x*n,y*n,3))
    else:
        img_array = np.zeros((x*n,y*n))

    # print (img_array.shape)
    idx = 0
    for i in range(n):
        for j in range(n):
            # print (i*x, (i+1)*x, j*y, (j+1)*y, idx)
            # imshow(image_batch[idx])
            if not label_image:
                img_array[i*x: (i+1)*x, j*y: (j+1)*y, :] = image_batch[idx]
            else:
                img_array[i*x: (i+1)*x, j*y: (j+1)*y] = image_batch[idx]
            idx=idx+1
    return (img_array)

def labelthreshold(image, threshold=0.5):
    label = np.zeros_like(image)
    label[image >= threshold] = 1
    return label

def normalize_minmax(data):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max-_min)!=0:
        img = (data - _min) / (_max-_min)
    else:
        img = np.zeros_like(data)            
    return img
    
# Image Helper Functions
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
    plt.show()


# Training Utils function
def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]
        
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))

def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded_ch1(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def softmax_dice_loss(y_true, y_pred):
    return (categorical_crossentropy(y_true, y_pred) * 0.5 \
    + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.25 \
    + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.25)

def softmax_dice_focal_loss(y_true, y_pred):
    return (binary_focal_loss(y_true, y_pred) * 0.5 \
    + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.25 \
    + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.25)

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed



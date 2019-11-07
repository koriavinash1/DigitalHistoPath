import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian

# Fully connected CRF post processing function
def do_crf(im, mask, n_labels, enable_color=False, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    # n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    if enable_color:
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP

def post_process_crf(image, final_probabilities, num_cl):
    softmax = final_probabilities.squeeze()
    softmax = softmax.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the unary_from_softmax for more information
    unary = unary_from_softmax(softmax, scale=None, clip=1e-5)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], num_cl)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                    img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return res 

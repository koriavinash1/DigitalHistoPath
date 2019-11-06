import numpy as np
from data_loader import DataGenerator
from imgaug import augmenters as iaa
from models import unet_densenet121
from utils import imshow


def predictor(data_dir, model_path):
    augmentation = iaa.SomeOf((0, 3), 
                [
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.Noop(),
                    iaa.OneOf([iaa.Affine(rotate=90),
                               iaa.Affine(rotate=180),
                               iaa.Affine(rotate=270)]),
                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                ])

    valid_transform_params = {'image_size': (256,256),
                              'batch_size': 32,
                              'n_classes': 2,
                              'n_channels': 3,
                              'shuffle': True,
                              'level': 'L0',
                              'transform': None
                             }
    # Generators
    # data_generator = DataGenerator(data_dir, **valid_transform_params)

    model = unet_densenet121((None, None), weights=None)
    model.load_weights(model_path)
    X = np.zeros((1,256,256,3))
    y_pred = model.predict(X, batch_size=1, verbose=0, steps=None)

    # for X, y in data_generator:
    # #     print (X.shape, y.shape)
    #     y_pred = model.predict(X, batch_size=1, verbose=0, steps=None)
    # #     print (y_pred.shape)
    #     imshow(X[0], y[0][:,:,1], y_pred[0][:,:,1])
    #     print (np.unique(y[0][:,:,1]), np.unique(y_pred[0][:,:,1]))

if __name__ == '__main__':
    # Model Prediction
    model_path = '../../saved_models/unet_densenet121_imagenet_pretrained_L0_20190711-182836/Model_Stage2.h5'
    # Training Data Configuration
    # Data Path
    train_data_dir = "D:\Coding_Projects\Camelyon\Camelyon_17_samples\train_dir"
    predictor(train_data_dir, model_path)
  
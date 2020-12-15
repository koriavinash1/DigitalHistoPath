# DigitalHistoPath
[![arXiv](https://img.shields.io/badge/arXiv-2001.00258-<COLOR>.svg)](https://arxiv.org/abs/2001.00258)

This repository contains the code for the cancer analysis framework proposed in [the paper](https://arxiv.org/abs/2001.00258) "A Generalized Deep Learning Framework for Whole-Slide Image Segmentation and Analysis"

## Brief Overview of Framework 
The framework consists of a segmentation algorithm optimized for histopathology tissue samples. A patch-based approach is utilized to break down the large size of these images.

It also has the code to empirically calculate the viable tumor burden. Viable tumor burden is the ratio of the viable tumor region to the whole tumor region. 

For more details, you can refer to our paper.

Our framework placed in several [grand-challenges](https://grand-challenge.org/challenges/):
| Challenge Name      | Description                    |  Position |
|---------------------|--------------------------------|-----------|
| [PAIP 2019](https://paip2019.grand-challenge.org/) - Task 1  | Segmentation of Liver Cancer   | 3<sup>rd</sup>       |
| [PAIP 2019 ](https://paip2019.grand-challenge.org/) - Task 2  | Viable Tumor Burden Estimation | 2<sup>nd</sup>       |
| [DigestPath 2019](https://digestpath2019.grand-challenge.org/Dataset/)     | Segmentation of Colon Cancer   | 4<sup>th</sup>       |

## Instructions
Kindly note that this repository has been made available as it was at the time of writing our paper. You may have to modify hard-coded parameters present in the code (data locations, image sizes etc) for the scripts to work for you. Kindly write to us if you have any difficulty in doing so. 

### Training
Training is divided into two stages: 
1. Extraction of patches - Patch coordinates are extracted randomly and stored in text files
1. Model training - The text files are used to train the models by generating the images on the fly


#### Patch extraction
The `points_extractor.py` under `code_cm17/patch_extraction` is responsible for this.

#### Model training
Run the `trainer.py` file present under `code_cm17/trainer` to train the three models.

### Inference
Edit the `CONFIG` dictionary in `code_cm17/inference/predict.py` and run the script.

## DigiPathAI
We packaged our inference pipeline into an full-fledged GUI application. Check it out [here](https://github.com/haranrk/DigiPathAI). It also contains our trained models for DigestPath and PAIP dataset.

## Contact 
- Avinash Kori (koriavinash1@gmail.com)
- Haran Rajkumar (haranrajkumar97@gmail.com)
- Mahendra Khened (mahendrakhened@gmail.com)

## Citation
If you find this reference implementation useful in your research, please consider citing:
```
@article{khened2020generalized,
  title={A Generalized Deep Learning Framework for Whole-Slide Image Segmentation and Analysis},
  author={Khened, Mahendra and Kori, Avinash and Rajkumar, Haran and Srinivasan, Balaji and Krishnamurthi, Ganapathy},
  journal={arXiv preprint arXiv:2001.00258},
  year={2020}
}
```

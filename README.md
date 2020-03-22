# DigitalHistoPath
This repository contains the code for the cancer analysis framework proposed in [this paper](https://arxiv.org/abs/2001.00258)

# Brief Overview of Framework 
The framework consists of a segmentation algorithm optimized for histopathology tissue samples. A patch-based approach is utilized to break down the large size of these images.

It also has the code to empirically calculate the viable tumor burden. Viable tumor burden is the ratio of the viable tumor region to the whole tumor region. 

For more details, you can refer to our paper.

Our framework placed in several [grand-challenges](https://grand-challenge.org/challenges/):
| Challenge Name      | Description                    |  Position |
|---------------------|--------------------------------|-----------|
| [PAIP 2019](https://paip2019.grand-challenge.org/) - Task 1  | Segmentation of Liver Cancer   | 3<sup>rd</sup>       |
| [PAIP 2019 ](https://paip2019.grand-challenge.org/) - Task 2  | Viable Tumor Burden Estimation | 2<sup>nd</sup>       |
| [DigestPath 2019](https://digestpath2019.grand-challenge.org/Dataset/)     | Segmentation of Colon Cancer   | 4<sup>th</sup>       |


### Refer [DigiPathAI](https://github.com/haranrk/DigiPathAI) for trained model and inference scripts 

### Citation
If you find this reference implementation useful in your research, please consider citing:
```
@article{khened2020generalized,
  title={A Generalized Deep Learning Framework for Whole-Slide Image Segmentation and Analysis},
  author={Khened, Mahendra and Kori, Avinash and Rajkumar, Haran and Srinivasan, Balaji and Krishnamurthi, Ganapathy},
  journal={arXiv preprint arXiv:2001.00258},
  year={2020}
}
```

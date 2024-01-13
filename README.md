# DC2Fusion

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2310.06291-red)](https://arxiv.org/abs/2310.06291)

This repo is the official implementation of [Three-Dimensional Medical Image Fusion with Deformable Cross-Attention](https://link.springer.com/chapter/10.1007/978-981-99-8141-0_41)

## Introduction
Multimodal medical image fusion plays an instrumental role
in several areas of medical image processing, particularly in disease recognition and tumor detection. Traditional fusion methods tend to process
each modality independently before combining the features and reconstructing the fusion image. However, this approach often neglects the
fundamental commonalities and disparities between multimodal information. Furthermore, the prevailing methodologies are largely confined
to fusing two-dimensional (2D) medical image slices, leading to a lack
of contextual supervision in the fusion images and subsequently, a
decreased information yield for physicians relative to three-dimensional
(3D) images. In this study, we introduce an innovative unsupervised feature mutual learning fusion network designed to rectify these limitations.
Our approach incorporates a Deformable Cross Feature Blend (DCFB)
module that facilitates the dual modalities in discerning their respective
similarities and differences. We have applied our model to the fusion of
3D MRI and PET images obtained from 660 patients in the Alzheimer’s
Disease Neuroimaging Initiative (ADNI) dataset. Through the application of the DCFB module, our network generates high-quality MRI-PET
fusion images. Experimental results demonstrate that our method surpasses traditional 2D image fusion methods in performance metrics such
as Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index
Measure (SSIM). Importantly, the capacity of our method to fuse 3D
images enhances the information available to physicians and researchers,
thus marking a significant step forward in the field. 


![network](https://github.com/louelin/DC2Fsuion/blob/main/imgs/network.png)
![network](https://github.com/louelin/DC2Fsuion/blob/main/imgs/Position_Relationship.png)


## ADNI Dataset
Due to restrictions, we cannot distribute our brain MRI. You can find the data we used in the 
[ADNI database](https://adni.loni.usc.edu/data-samples/access-data/),And you can find the image numbers we used in the ImageID.csv file, which can be obtained through the ADNI advanced search function.We performed simple processing of the data, including conversion of dcm files to nii format using [pydicom](https://github.com/pydicom/pydicom), and simple alignment of PET to MRI images using the [ANTsPy](https://github.com/ANTsX/ANTsPy) library.

## Train

```cmd
python train_diff.py --alpha1 2 --alpha2 1 --beta1 2 --beta2 1 --gamma1 2 --gamma2 1 --gpus 0 
```
## Test

```cmd
python infer_sigleImage.py
```
![result](https://github.com/louelin/DC2Fsuion/blob/main/imgs/result.png)

## Citation
If you find this code is useful in your research, please consider to cite:

```
@InProceedings{10.1007/978-981-99-8141-0_41,
author="Liu, Lin and Fan, Xinxin and Zhang, Chulong and Dai, Jingjing and Xie, Yaoqin and Liang, Xiaokun",
title="Three-Dimensional Medical Image Fusion with Deformable Cross-Attention",
booktitle="Neural Information Processing",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="551--563",
isbn="978-981-99-8141-0"
}

```

## Reference:

[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main
)

[U2Fusion](https://github.com/hanna-xu/U2Fusion)

[SwinFuse](https://github.com/Zhishe-Wang/SwinFuse)

[Dilran](https://github.com/simonZhou86/dilran)


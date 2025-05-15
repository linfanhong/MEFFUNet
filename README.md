<div align="center">
<h1 align="center">DSFormer</h1>

<h3>A Multi-Scale Efficient Feature Fusion U-Net Model for Spinal Image Segmentation![image](https://github.com/user-attachments/assets/1dd212e9-d275-4ac4-b47e-2cfeee047c7d)
</h3>

[Yichu Xu](https://scholar.google.com/citations?user=CxKy4lEAAAAJ&hl=en&oi=ao)<sup>1</sup>, 
[Di Wang](https://scholar.google.com/citations?user=3fThjewAAAAJ&hl=en)<sup>1</sup>, 
[Lefei Zhang](https://scholar.google.com/citations?user=BLKHwNwAAAAJ&hl=en)<sup>1 *</sup>, 
[Liangpei Zhang](https://scholar.google.com/citations?user=vzj2hcYAAAAJ&hl=en)<sup>1,2 </sup>

[![NN paper](https://img.shields.io/badge/NN-paper-00629B.svg)](https://www.sciencedirect.com/science/article/pii/S089360802500190X?dgcid=author1) [![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2410.03171)

<sup>1</sup> Wuhan University, <sup>2</sup> Henan Academy of Sciences,  <sup>*</sup> Corresponding author

</div>


## 📖Overview

* [**DSFormer**](https://arxiv.org/abs/2410.03171) is a novel Dual Selective Fusion
Transformer Network  for HSI classification. It adaptively selects and fuses features from diverse
receptive fields to achieve joint spatial-spectral context modeling, while reducing unnecessary information
interference by focusing on the most relevant spatial-spectral tokens.  

<div align="center">
  <img src="./figures/DSFormer.png"><br><br>
</div>

## 🚀Let's Get Started!
### `A. Installation`
**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/YichuXu/DSFormer.git
cd MEFFUNet
```

**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n MEFFUNet
conda activate MEFFUNet
```

***Install dependencies***

Our method uses python 3.8, pytorch 1.13

### `B. Data Preparation`

Download HSI classification dataset from [Google Drive](https://drive.google.com/drive/folders/1iPFLdrAFUveqwCtMpf5859pQhGXN_z4J?usp=drive_link) or [Baidu Drive (百度网盘)](https://pan.baidu.com/s/1bSqq-Uv3AC5qfRmqxbMjfg?pwd=2025) and put it under the [dataset] folder. It will have the following structure: 
```
${DATASET_ROOT}   # Dataset root directory
├── datasets
│   │
│   ├── pu        # Pavia University data
│   │   ├──images
│   │   ├──masks
│   │
│   ├── houston13  # Houston 2013 data
│   │   ├──images
│   │   ├──masks 

```

### `C. Performance Evaluation`
- The following commands show how to train and evaluate DSFormer for HSI classification:
```bash

```

## 📜Reference

if you find it useful for your research, please consider giving this repo a ⭐ and citing our paper! We appreciate your support！😊
```
@ARTICLE{Xu2025DSFormer,
  author={},
  title={A Multi-Scale Efficient Feature Fusion U-Net Model for Spinal Image Segmentation![image](https://github.com/user-attachments/assets/083f3947-884b-4bf3-8e95-0e650fb9c52f)
}, 
  journal={},
  volume = {},
  pages = {},
  year = {2025}
}
```

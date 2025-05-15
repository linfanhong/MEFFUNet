<div align="center">
<h1 align="center">MEFFUNet</h1>

<h3>A Multi-Scale Efficient Feature Fusion U-Net Model for Spinal Image Segmentation
</h3>

[]()<sup>1</sup>, 
[]()<sup>1 *</sup>, 
[]()<sup>1,2 </sup>


School of Integrated Circuit Science and Engineering, Tianjin University of Technology, Tianjin 300384, China
</div>


## 📖Overview

* [**MEFFUNet**]() is a lightweight and efficient U-Net variant tailored for spinal image segmentation, which strikes a balance between segmentation accuracy and computational efficiency through three core theoretical and architectural innovations. Extensive experiments on both MRI and CT datasets demonstrate the superior performance and generalizability of MEFFUNet, outperforming established baselines in segmentation accuracy with considerably fewer parameters.



<div align="center">
  <img src="./Figures/MEFFUNet.svg"><br><br>
</div>

## 🚀Let's Get Started!
### `A. Installation`
**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/linfanhong/MEFFUNet.git
cd MEFFUNet
```

**Step 2: Environment Setup:**

It is recommended to set up a conda environment and install dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n MEFFUNet
conda activate MEFFUNet
```

***Install dependencies***

Our method uses python 3.8, pytorch 1.13

### `B. Data Preparation`

Download SpineSagT2Wdataset3 and TotalSegmentator dataset from [Baidu Drive (百度网盘)](https://pan.baidu.com/s/1_N9v9UWWArPbq3h0oqhZ5Q) or [zenodo](https://zenodo.org/records/10047292) and put them under the [inputs] folder. It will have the following structure: 
```
${DATASET_ROOT}   # Dataset root directory
├── inputs
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

If you find it useful for your research, please consider giving this repo a ⭐ and citing our paper! We appreciate your support！😊
```
@ARTICLE{Xu2025DSFormer,
  author={},
  title={A Multi-Scale Efficient Feature Fusion U-Net Model for Spinal Image Segmentation
}, 
  journal={},
  volume = {},
  pages = {},
  year = {2025}
}
```

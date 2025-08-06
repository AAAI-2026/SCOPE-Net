<div align="center">
<h1>SCOPE-Net: A Structure-aware and Context-cooperative Network for Salient Object Detection in Optical Remote Sensing Images </h1>
</div>

## 📰 News
This project provides the code and results for DASGNet.

## ⭐ Abstract
Salient object detection in optical remote sensing images (ORSI-SOD) aims to accurately segment the most visually prominent ground objects from complex aerial scenes. However, existing methods still struggle to balance global semantic discrimination and local detail precision, often failing to handle structural complexity and scale variation. Moreover, traditional boundary enhancement lacks global context guidance, resulting in blurred edges and incomplete contours. To address this, we propose SCOPE-Net, a Structure-aware and COntext-cooPErative Network with two key modules: a Structure-aware Attention Module (SAM) for structural modeling, and a Dynamic Context-integrated Edge Refinement Module (DCERM) for boundary enhancement. Specifically, SAM models direction-aware spatial long- and short-range dependencies and integrates multi-granularity channel features to improve adaptability to structural complexity and scale variation, enabling collaborative representation of global semantics and local details. Meanwhile, DCERM dynamically fuses edge information and high-level semantics by adaptively extracting multi-scale boundary features and applying a dual-path fusion strategy, effectively enhancing contour integrity and localization accuracy. Extensive experiments on three ORSI-SOD benchmarks show that SCOPE-Net outperforms 14 state-of-the-art methods, reducing $\mathcal{M}$ by 9.8\%, improving $\mathit{S}_{\alpha}$ by up to 0.73\%, and $F_{\beta}^{\text{max}}$ by 0.52\%. The code is publicly available at: https://github.com/AAAI-2026/SCOPE-Net.

## 🌏 Network Architecture
   <div align=center>
   <img src="https://github.com/AAAI-2026/SCOPE-Net/blob/main/SCOPE-Net/images/SCOPE-Net.png">
   </div>
Overview of SCOPE-Net. It consists of two key modules: a Structure-aware Attention Module (SAM) and a Dynamic Context-integrated Edge Refinement Module (DCERM).



<div align=center>
   <img src="https://github.com/AAAI-2026/SCOPE-Net/blob/main/SCOPE-Net/images/SAM.png">
   </div>
Illustrations of the proposed SAM.

<div align=center>
   <img src="https://github.com/AAAI-2026/SCOPE-Net/blob/main/SCOPE-Net/images/DCERM.png">
   </div>
Illustrations of the proposed DCERM.
   
## 🖥️ Requirements
   python 3.8 + pytorch 1.9.0
   
## 🚀 Training
   Download [pvt_v2_b2.pth] and put it in './model/'. 
   
   Modify paths of datasets, then run train.py.

Note: Our main model is under './model/DASGNet.py'

## 🛸 Testing
   1. Modify paths of pre-trained models and datasets.

   2. Run test.py.

## 🖼️ Quantitative comparison
   <div align=center>
   <img src="https://github.com/AAAI-2026/SCOPE-Net/blob/main/SCOPE-Net/images/table.png">
   </div>
   
## 🌃 Visualization
   <div align=center>
   <img src="https://github.com/AAAI-2026/SCOPE-Net/blob/main/SCOPE-Net/images/Visualization.png">
   </div>

# PorkBellyHSI
![](graphical_abstract.png)

This repository contains source code showing the model architecture, loss function, and training pipeline used by Engstrøm et al. [[1]](#references) to generate chemical maps of pork bellies with a modified U-Net [[2]](#references). If you are interested in a U-Net implementation, [this repository](https://github.com/sm00thix/unet) releases a U-Net implementation under the permissive Apache 2.0 License.

The weights for the ensemble of five modified U-Nets used by CITE is available on HuggingFace (LINK).


## References
1. [O.-C. G. Engstrøm, M. Albano-Gaglio, E. S. Dreier, Y. Bouzembrak, M. Font-i-Furnols, P. Mishra, and K. S. Pedersen (2025). Transforming Hyperspectral Images Into Chemical Maps: An End-to-End Deep Learning Approach](https://arxiv.org/abs/2504.14131)

2. [O. Ronneberger, P. Fischer, and Thomas Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.](https://arxiv.org/abs/1505.04597)

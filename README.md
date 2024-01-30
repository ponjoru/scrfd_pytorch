# SCRFD_PyTorch
This is a pure pytorch re-implementation of the SCRFD face detector by insightface [original repo](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

## Note:
The metrics provided by the current repo are slightly different due to the difference in post-processing.
The models are tested to output the same raw result.

## Motivation
* The original implementation of SCRFD is based on mmdetection framework which is not quite flexible
* The pytorch-only implementation is a way easier to integrate into your own pipeline avoiding digging into a mmdet dependency hell
* The current version provides a more clean and easy-to-understand code compared to mmdet implementation
* The Mmdet implementation is not TorchScript-friendly

## The Repo contains:
- [x] Converted weights Vanilla to Current (the conversion script is attached)
- [x] Reimplemented SCRFD models
- [x] Refactored Loss computation
- [x] Improved BboxSafeRandomCrop augmentation proposed in the original SCRFD paper, implemented with the albumentations library
- [ ] TorchScript conversion and inference example
- [ ] Onnx conversion and inference example
- [ ] Evaluation results matrix

## Getting started
tbd

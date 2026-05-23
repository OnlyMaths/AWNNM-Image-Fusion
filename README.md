# Adaptive Weighted Nuclear Norm Minimization (AWNNM) for Image Fusion

Official implementation of the paper:  
**"Provably Convergent Low-Rank Grayscale Image Fusion via Adaptive Weighted Nuclear Norm Minimization"**

## Overview

This repository contains the PyTorch implementation of **Adaptive WNNM**, a fully unsupervised, training-free, and provably convergent method for multi-modal grayscale image fusion.

### Key Features
- Globally convergent with theoretical guarantees
- Only **one** tunable parameter (`λ`)
- Efficient using randomized SVD
- No training data or GPU required
- Excellent performance on TNO, RoadScene, and Medical datasets

## Installation

```bash
git clone https://github.com/gargitrivedi/AWNNM-Image-Fusion.git
cd AWNNM-Image-Fusion
pip install -r requirements.txt


## Citation

If you use this work, please cite:

@article{trivedi2026awnnm,
  title={Adaptive Weighted Nuclear Norm Minimization for Image Fusion},
  author={Trivedi, Gargi},
  year={2026}
}

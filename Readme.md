# DIVE-Net

Official code for the paper:

**DIVE-Net: Depthwise Multi-Dilation Vision Network for Efficient Underwater Image Enhancement**
Sourav Phogat, Koyyada Dinesh Kumar, Sujit Kumar Sahoo
AUSTech 2026 Workshop (Advances in Underwater Surveillance: Technologies, Challenges, and Future Directions), ICPR 2026, Lyon

## Results

On the UIEB test set, as reported in the paper: PSNR 22.32 dB, SSIM 0.893, with 0.74M parameters.

## Requirements

```
pip install torch torchvision pyyaml pillow tqdm thop numpy opencv-python
```

A CUDA GPU is needed to run `test.py`.

## Dataset

Download the UIEB dataset and place the test images as:

```
Dataset/UIEB/input/    # raw underwater images
Dataset/UIEB/target/   # reference images
```

Input and target images must have the same file names.

## Pretrained models

Checkpoints are in `checkpoints/DIVENet_Checkpoints/models/`.

## Testing

```
python test.py
```

This loads `model_bestPSNR.pth`, prints PSNR/SSIM for each image and the mean, and saves the outputs to `checkpoints/DIVENet_Checkpoints/test_results/`. Settings can be changed in `test.yaml`.

## Citation

```
@inproceedings{phogat2026divenet,
  title     = {DIVE-Net: Depthwise Multi-Dilation Vision Network for Efficient Underwater Image Enhancement},
  author    = {Phogat, Sourav and Koyyada, Dinesh Kumar and Sahoo, Sujit Kumar},
  booktitle = {International Conference on Pattern Recognition (ICPR) Workshops},
  year      = {2026}
}
```

import torch
import cv2
from pytorch_msssim import ssim
from torchvision.transforms.functional import pad
import torch.nn.functional as f
import numpy as np

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def torchSSIM(tar_img, prd_img):
    #print(tar_img.shape)
    #print(prd_img.shape)
    _,_,H,W = tar_img.shape

    #pred = model(prd_img)[2]
    #pred = pred[:,:,:h,:w]

    #elapsed = time.time() - tm
    #adder(elapsed)

    pred_clip = torch.clamp(prd_img, 0, 1)

    pred_numpy = pred_clip.squeeze(0).cpu().numpy()
    label_numpy = tar_img.squeeze(0).cpu().numpy()


    label_img = (tar_img).cuda()


    down_ratio = max(1, round(min(H, W) / 256))
    ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)
    return ssim_val


def getUCIQE2(loc):
    #img_bgr = cv2.imread(loc)        # Used to read image files
    loc = torch.clamp(loc, 0, 1)
    loc = loc[0].detach().cpu().permute(1, 2, 0).numpy()
    #loc = (loc * 255).astype(np.uint8)
    img_lab = cv2.cvtColor(loc, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    uciqe = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    print(uciqe)
    return uciqe


import cv2
import numpy as np

def getUCIQE2mm(loc):
    loc = loc[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    loc = np.clip(loc, 0, 1)

    img_lab = cv2.cvtColor(loc, cv2.COLOR_RGB2LAB)
    print(img_lab)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576

    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(img_a**2 + img_b**2)

    eps = 1e-12
    img_sat = img_chr / np.sqrt(img_chr**2 + img_lum**2 + eps)

    aver_sat = np.mean(img_sat)
    var_chr = np.std(img_chr)
    con_lum = np.percentile(img_lum, 99) - np.percentile(img_lum, 1)

    uciqe = c1 * var_chr + c2 * con_lum + c3 * aver_sat
    #print(uciqe)
    return float(uciqe)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def pad_img(img, padded_size):
    C, H, W = img.shape
    pad_size = [(padded_size[0]-H) // 2, (padded_size[1]-W) // 2]
    return pad(img, pad_size)

	

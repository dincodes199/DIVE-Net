import os
import torch
import torchvision
import yaml
from PIL import Image
from utils.data_loader import get_test_data
from torch.utils.data import DataLoader
from utils.dir_utils import mkdir, get_last_path
from utils.model_utils import load_checkpoint
from model.main_model import DIVENet_Main as URSCT
from tqdm import tqdm
from utils.image_utils import torchPSNR, torchSSIM
import torchvision.transforms.functional as TF
from thop import profile
import numpy as np
import cv2
import numpy as np
import torch.nn.functional as F


def compute_mse(restored, target):
    return torch.mean((restored - target) ** 2)


def ssim(prediction, target):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = np.array(target, dtype=np.float64)
    #img2 = np.array(ref, dtype=np.float64)
    img1 = target.detach().cpu().numpy().astype(np.float64)
    img2 = ref.detach().cpu().numpy().astype(np.float64)
    img1 = np.transpose(img1[0], (1,2,0))
    img2 = np.transpose(img2[0], (1,2,0))
    #print(img1.shape)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def get_dataloader(opt_test):
    test_dataset = get_test_data(opt_test['TEST_DIR'], {'patch_size': opt_test['TEST_PS']})
    return DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=opt_test['NUM_WORKS'],
                      drop_last=False)

def compute_mse(restored, target):
    return torch.mean((restored - target) ** 2)

def inference_full_image_no_overlap(model, input_image, patch_size=256, device='cuda'):
    """
    No-overlap patch-based inference.
    """
    _, C, H, W = input_image.shape

    # compute padding so H,W divisible by patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    inp = F.pad(input_image, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hpad, Wpad = inp.shape

    # prepare output buffer
    output = torch.zeros_like(inp, device=device)

    # tile with stride = patch_size (i.e. no overlap)
    for y in range(0, Hpad, patch_size):
        for x in range(0, Wpad, patch_size):
            patch = inp[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                out_patch = model(patch.to(device))
                if isinstance(out_patch, (list, tuple)):
                    out_patch = out_patch[-1]
            output[:, :, y:y+patch_size, x:x+patch_size] = out_patch

    # crop back to original size
    return output[:, :, :H, :W]

def inference_full_image_overlap(model, input_image,
                                 patch_size=256,
                                 overlap=0.5,
                                 device='cuda'):
    """
    Sliding-window inference with overlap and averaging.
    overlap: fraction of patch_size to overlap (0 < overlap < 1).
    """
    B, C, H, W = input_image.shape
    stride = int(patch_size * (1 - overlap))
    # pad so that (Hpad, Wpad) are multiples of stride
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    inp = F.pad(input_image, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hpad, Wpad = inp.shape

    # buffers for accumulation
    output = torch.zeros_like(inp, device=device)
    weight = torch.zeros_like(inp, device=device)

    # sliding window
    for y in range(0, Hpad - patch_size + 1, stride):
        for x in range(0, Wpad - patch_size + 1, stride):
            patch = inp[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                out_patch = model(patch.to(device))
                if isinstance(out_patch, (list, tuple)):
                    out_patch = out_patch[-1]
            output[:, :, y:y+patch_size, x:x+patch_size] += out_patch
            weight[:, :, y:y+patch_size, x:x+patch_size] += 1.0

    # normalize and crop
    output = output / weight
    return output[:, :, :H, :W]



def predict_save_eval(test_loader, opt_test):

    PSNRs, SSIMs, mse, fsim, fsimc, lpips = [], [], [], [], [], []

    for i, data in enumerate(tqdm(test_loader)):

        input = data[0].to(device)
        target_SR = data[1].to(device)

        # Get image name from dataset paths
        img_name = os.path.basename(test_loader.dataset.inp_filenames[i])

        with torch.no_grad():
            restored_SR = model(input)[2]

        # Normalize images to [-1,1] for LPIPS
        x_tensor = (target_SR * 2) - 1
        x_net_tensor = (restored_SR * 2) - 1

        psnr11 = torchPSNR(restored_SR, target_SR)
        ssim11 = calculate_ssim(target_SR, restored_SR)

        print(f"Image: {img_name} | PSNR: {psnr11:.4f} | SSIM: {ssim11:.4f}")

        PSNRs.append(psnr11)
        SSIMs.append(ssim11)

        mse.append(compute_mse(restored_SR, target_SR))

        torchvision.utils.save_image(
            restored_SR,
            os.path.join(result_dir, str(i) + '.png')
        )

    return PSNRs, SSIMs, mse


if __name__ == '__main__':
    with open('test.yaml', 'r') as config:
        opt = yaml.safe_load(config)
        opt_test = opt['TEST']
    mode = opt_test['MODE']
    device = opt_test['DEVICE']
    model_detail_opt = opt['MODEL_DETAIL']
    result_dir = os.path.join(opt_test['SAVE_DIR'], opt['TRAINING']['MODEL_NAME'], 'test_results')
    mkdir(result_dir)

    
    model = URSCT().to(device)
    path_chk_rest = get_last_path(os.path.join(opt_test['SAVE_DIR'], opt['TRAINING']['MODEL_NAME'], 'models'), '_bestPSNR.pth')
    print(path_chk_rest)
    load_checkpoint(model, path_chk_rest)
    model.eval()

    x = torch.randn(1, 3, 256, 256).cuda()  # set your real input size
    macs, params = profile(model, inputs=(x,), verbose=False)

    flops = 2 * macs                 # if you count FLOPs = 2*MACs
    gflops_per_infer = flops / 1e9

    print("MACs:", macs)
    print("FLOPs:", flops)
    print("GFLOPs per inference:", gflops_per_infer)
    print("Params:", params)
    print("Params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    macs, _ = profile(model, inputs=(x,), verbose=False)
    true_params = sum(p.numel() for p in model.parameters())#

    print(f"FLOPs: {macs/1e9:.2f}G")          # matching his convention
    print(f"Params: {true_params/1e6:.2f}M")


    test_loader = get_dataloader(opt_test)
    
    PSNRs, SSIMs,mse = predict_save_eval(test_loader, opt_test)

    print("[PSNR] mean: {:.4f} std: {:.4f}".format(torch.stack(PSNRs).mean().item(), torch.stack(PSNRs).std().item()))
    print("[SSIM] mean: {:.4f} std: {:.4f}".format(torch.stack(SSIMs).mean().item(), torch.stack(SSIMs).std().item()))
    print("[mse] mean: {:.4f} std: {:.4f}".format(torch.stack(mse).mean().item(), torch.stack(mse).std().item()))
    print("[PSNR] mean: {:.4f} std: {:.4f}".format(
    np.mean([p.item() if torch.is_tensor(p) else p for p in PSNRs]),
    np.std([p.item() if torch.is_tensor(p) else p for p in PSNRs])
    ))

    print("[SSIM] mean: {:.4f} std: {:.4f}".format(
    np.mean(SSIMs),
    np.std(SSIMs)
    ))

    print("[mse] mean: {:.6f} std: {:.6f}".format(
    np.mean([m.item() if torch.is_tensor(m) else m for m in mse]),
    np.std([m.item() if torch.is_tensor(m) else m for m in mse])
    ))


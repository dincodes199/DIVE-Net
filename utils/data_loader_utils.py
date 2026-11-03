import torch
import os
import random
import shutil
from utils.dir_utils import mkdir
import natsort

class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

def Random_Split(origin_path, test_num, test_path):
    assert os.path.exists(origin_path)
    mkdir(os.path.join(test_path, 'input'))
    mkdir(os.path.join(test_path, 'target_enh'))
    inp_files = natsort.natsorted(os.listdir(os.path.join(origin_path, 'input')))
    choosed_inp_files = random.sample(inp_files, test_num)
    for i in choosed_inp_files:
        origin_inp_path = os.path.join(origin_path, 'input', i)
        origin_tag_path = os.path.join(origin_path, 'target_enh', i)
        move_inp_path = os.path.join(test_path, 'input', i)
        move_tag_path = os.path.join(test_path, 'target_enh', i)
        shutil.move(origin_inp_path, move_inp_path)
        shutil.move(origin_tag_path, move_tag_path)


import numpy as np
import torch
import torch.fft as fft
import math


def unnormalize(x):
    # restore from T.Normalize
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # self.mean = np.array(torch.tensor(mean).view((-1, 1, 1)))
    # self.std = np.array(torch.tensor(std).view((-1, 1, 1)))
    device = x.device
    mean = torch.tensor(mean).view((-1, 1, 1)).to(device)
    std = torch.tensor(std).view((-1, 1, 1)).to(device)
    # print(x.shape)
    x = x * std + mean

    return torch.clip(x, 0, None)  # torch.clip(input, min=None, max=None) → Tensor


def DFTImg(img, patch_size=8):
    height, width = img[0].shape[-2], img[0].shape[-1]
    lpf = torch.zeros((height, width))
    R = (height + width) // patch_size
    for x in range(width):
        for y in range(height):
            if ((x - (width - 1) / 2) ** 2 + (y - (height - 1) / 2) ** 2) < (R ** 2):
                # print(True)
                lpf[y, x] = 1

    hpf = 1 - lpf
    lpf = lpf.unsqueeze(dim=-1)
    hpf = hpf.unsqueeze(dim=-1)
    img_dft = img.clone()
    imgl = img.clone()
    imgh = img.clone()
    device = img.device
    for index in range(img_dft.shape[0]):
        x = img_dft[index]
        imgx = unnormalize(x)
        imgx = fft.fftn(imgx, dim=(0, 1))
        imgx = torch.roll(imgx, (height // 2, width // 2),
                          dims=(0, 1))  # torch.roll(input, shifts, dims=None) → Tensor
        # print(imgx.shape, lpf.shape)
        img_l = imgx * lpf.view((lpf.shape[0], lpf.shape[1])).to(device)
        img_h = imgx * hpf.view((lpf.shape[0], lpf.shape[1])).to(device)
        imgl[index] = torch.abs(fft.ifftn(img_l, dim=(0, 1)))
        imgh[index] = torch.abs(fft.ifftn(img_h, dim=(0, 1)))

    return imgl, imgh


def DFTFeatureMap(fm, patch_size=8):
    height, width = fm[0].shape[-2], fm[0].shape[-1]
    attention = fm.clone()

    device = fm.device
    for index in range(fm.shape[0]):
        x = fm[index]
        # fmx = unnormalize(x)
        fmx = fft.fftn(x, dim=(1, 2), norm='ortho')
        # fmx = torch.roll(fmx, (height // 2, width // 2),
        #                  dims=(1, 2))
        fmx_conj = torch.conj(fmx)
        attention[index] = fmx * fmx_conj
    F_FA = fm + 0.01 * fft.ifftn(attention, dim=(1, 2), norm='ortho').abs()*fm
    # F_FA = fm + 0.01 * attention

    F_FO = fm * torch.sum(attention) * torch.norm(fm) * torch.norm(fm)

    feature_map = F_FA + F_FO
    # feature_map = F_FA

    return feature_map


def cal_losses(losses_o, losses_l, losses_h):
    losses = dict()
    for i in range(len(losses_h.keys())):
        lok = list(losses_o.keys())
        llk = list(losses_l.keys())
        lhk = list(losses_h.keys())
        key_o = str(lok[i] + '_o')
        key_l = str(llk[i] + '_l')
        key_h = str(lhk[i] + '_h')
        losses[key_o] = losses_o[lok[i]]
        losses[key_l] = losses_l[llk[i]]
        losses[key_h] = losses_h[lhk[i]]
    return losses

def cal_losses_l(losses_o, losses_l):
    losses = dict()
    for i in range(len(losses_l.keys())):
        lok = list(losses_o.keys())
        llk = list(losses_l.keys())
        key_o = str(lok[i] + '_o')
        key_l = str(llk[i] + '_l')
        losses[key_o] = losses_o[lok[i]]
        losses[key_l] = losses_l[llk[i]]
    return losses

def cal_losses_h(losses_o,  losses_h):
    losses = dict()
    for i in range(len(losses_h.keys())):
        lok = list(losses_o.keys())
        lhk = list(losses_h.keys())
        key_o = str(lok[i] + '_o')
        key_h = str(lhk[i] + '_h')
        losses[key_o] = losses_o[lok[i]]
        losses[key_h] = losses_h[lhk[i]]
    return losses
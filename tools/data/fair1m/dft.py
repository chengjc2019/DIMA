import numpy
import numpy as np
import torch
import torch.fft as fft
import os
import mmcv
import copy
import cv2

def unnormalize(x):
    # restore from T.Normalize
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # self.mean = np.array(torch.tensor(mean).view((-1, 1, 1)))
    # self.std = np.array(torch.tensor(std).view((-1, 1, 1)))
    # device = x.device
    # mean = torch.tensor(mean).view((-1, 1, 1)).to(device)
    # std = torch.tensor(std).view((-1, 1, 1)).to(device)
    # x = x * std + mean
    # print(x.shape)
    mean = np.array(torch.tensor(mean).view((1, 1, -1)))
    std = np.array(torch.tensor(std).view((1, 1, -1)))
    # x = (x * self.std) + self.mean
    x = torch.tensor(x) * std + mean
    return torch.clip(x, 0, None)


def DFTImg(img, patch_size=8):
    height, width = img.shape[:2]
    lpf = torch.zeros((height, width))
    R = (height + width) // patch_size
    for x in range(width):
        for y in range(height):
            if ((x - (width - 1) / 2) ** 2 + (y - (height - 1) / 2) ** 2) < (R ** 2):
                lpf[y, x] = 1

    hpf = 1 - lpf
    lpf = lpf.unsqueeze(dim=-1)
    hpf = hpf.unsqueeze(dim=-1)
    # device = img.device
    imgx = unnormalize(copy.deepcopy(img))
    imgx = fft.fftn(imgx, dim=(0, 1))
    imgx = torch.roll(imgx, (height // 2, width // 2), dims=(0, 1))  # torch.roll(input, shifts, dims=None) → Tensor
    # print(imgx.shape, lpf.shape)
    f_imgx = 20*np.log(np.abs(np.array(imgx)))

    imgl = imgx * lpf
    imgh = imgx * hpf
    # cv2.imshow(imgl)
    f_imgl = 20*np.log(np.abs(np.array(imgl)))
    f_imgh=20*np.log(np.abs(np.array(imgh)))
    imgl = torch.abs(fft.ifftn(imgl, dim=(0, 1)))
    imgh = torch.abs(fft.ifftn(imgh, dim=(0, 1)))

    return imgl, imgh,f_imgx,f_imgl,f_imgh


# for index in range(img_dft.shape[0]):
#         x = img_dft[index]
#         imgx = unnormalize(x)
#         imgx = fft.fftn(imgx, dim=(0, 1))
#         imgx = torch.roll(imgx, (height // 2, width // 2),
#                           dims=(0, 1))  # torch.roll(input, shifts, dims=None) → Tensor
#         # print(imgx.shape, lpf.shape)
#         img_l = imgx * lpf.view((lpf.shape[0], lpf.shape[1])).to(device)
#         img_h = imgx * hpf.view((lpf.shape[0], lpf.shape[1])).to(device)
#         imgl[index] = torch.abs(fft.ifftn(img_l, dim=(0, 1)))
#         imgh[index] = torch.abs(fft.ifftn(img_h, dim=(0, 1)))
#
#     return imgl, imgh


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


ori_img_files = '/home/chip/datasets/FAIR1M/raw/train/images/'
dft_img_files = '/media/chip/A26A0E196A0DEABD/work_dirs/vis/dft_images/'

for img_file in os.listdir(ori_img_files):
    name = img_file.split('.')[0]
    img = mmcv.imread(os.path.join(ori_img_files, img_file))
    # img = torch.tensor(img).to(device='cuda:0')
    # img = img.permute([2,0,1])
    img_l, img_h,f_imgx,f_imgl,f_imgh = DFTImg(img)

    save_file_l = os.path.join(dft_img_files, 'low', str(name + '.png'))
    mmcv.imwrite(numpy.array(img_l), save_file_l)

    save_file_h = os.path.join(dft_img_files, 'high', str(name + '.png'))
    mmcv.imwrite(numpy.array(img_h), save_file_h)

    save_file_fx = os.path.join(dft_img_files, 'fx', str(name + '.png'))
    mmcv.imwrite(numpy.array(f_imgx), save_file_fx)

    save_file_fl = os.path.join(dft_img_files, 'fl', str(name + '.png'))
    mmcv.imwrite(numpy.array(f_imgl), save_file_fl)

    save_file_fh = os.path.join(dft_img_files, 'fh', str(name + '.png'))
    mmcv.imwrite(numpy.array(f_imgh), save_file_fh)

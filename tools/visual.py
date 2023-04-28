import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def heatmap(heat):
    channel = 160
    heat = heat[0].cpu().numpy()
    heat = np.squeeze(heat, 0)
    heat = heat[channel:channel + 3, :, :]
    heatmap = np.maximum(heat, 0)
    heatmap = np.mean(heatmap, axis=0)
    heatmap /= np.max(heatmap)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # plt.imshow(heatmap)
    # plt.show()
    # cv2.imwrite("/media/chip/A26A0E196A0DEABD/work_dirs/fair1m/dft_vis/dft_features/ori/6437.png", heatmap)
    cv2.imwrite("/media/chip/A26A0E196A0DEABD/work_dirs/fair1m/dft_vis/dft_features/low/6437.png", heatmap)
    # cv2.imwrite("/media/chip/A26A0E196A0DEABD/work_dirs/fair1m/dft_vis/dft_features/high/6437.png", heatmap)

    # cv2.imwrite("/media/chip/A26A0E196A0DEABD/work_dirs/fair1m/test_vis/test_features/1419.png", heatmap)
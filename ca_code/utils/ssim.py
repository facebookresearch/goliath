# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Project code: https://github.com/Po-Hsun-Su/pytorch-ssim
# Copyright Evan Su, 2017
# License: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt (MIT)

import torch as th
import torch.nn.functional as thf
from math import exp

def gaussian(window_size, sigma):
    gauss = th.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = thf.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = thf.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = thf.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = thf.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = thf.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        mask = mask.expand(-1, ssim_map.shape[1], -1, -1)
        ssim_map = ssim_map * mask

    if size_average:
        if mask is not None:
            return ssim_map.sum() / mask.sum().clamp(min=1)
        else:
            return ssim_map.mean()
    else:
        if mask is not None:
            return ssim_map.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1)
        else:
            return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.shape[-3]
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

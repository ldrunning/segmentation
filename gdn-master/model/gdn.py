import torch
from torch import nn
import torch.nn.functional as F
# from model/resnet import ResNet as models
import model.resnet as models
# import resnet as models
# import resnet as models
from torch.nn.utils import weight_norm
import time
from math import exp
import numpy as np


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# from .common import ShiftMean


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.pool = nn.AvgPool2d(2, 2)
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        gw1 = self.gw1(x)
        gw2 = self.gw2(x)
        gw = torch.cat([gw1, gw2], dim=1)
        gw = self.re(gw)
        return x + self.module(gw) * self.res_scale


class ResBlock_2(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):
        super(ResBlock_2, self).__init__()
        self.res_scale = res_scale
        self.pool = nn.AvgPool2d(2, 2)
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        x1 = self.pool(x)
        gw1 = self.gw1(x1)
        gw2 = self.gw2(x1)
        gw = torch.cat([gw1, gw2], dim=1)
        gw = self.re(gw)
        gw = self.module(gw) * self.res_scale
        gw = nn.functional.interpolate(gw, scale_factor=2, mode='bilinear', align_corners=True)
        return x + gw


class Body_1(nn.Module):
    def __init__(self):
        super(Body_1, self).__init__()
        w_residual = [ResBlock(54, 6, 1.0, 0.75)
                      for _ in range(6)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x1 = self.module(x)
        return x + x1


class Body_2(nn.Module):
    def __init__(self):
        super(Body_2, self).__init__()
        w_residual = [ResBlock_2(54, 6, 1.0, 0.75)
                      for _ in range(6)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x, y):
        x1 = self.module(x)
        x = x + x1
        return torch.cat([x, y], dim=1)


class ResBlock_3(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock_3, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        return p1 + p2


class Body_3(nn.Module):
    def __init__(self):
        super(Body_3, self).__init__()
        self.con = weight_norm(nn.Conv2d(3, 30, kernel_size=3, padding=1))
        w_residual = [ResBlock_3(30)
                      for _ in range(3)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x = self.con(x)
        x = self.module(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class GDN(nn.Module):
    def __init__(self, layers=12, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=False):
        super(GDN, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        head = [weight_norm(nn.Conv2d(30, 54, kernel_size=3, padding=1))]
        tail = [nn.Upsample(scale_factor=4, mode='nearest'),
                weight_norm(nn.Conv2d(108, 108, kernel_size=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(108, 32, kernel_size=3, padding=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(32, 20, kernel_size=3, padding=1)),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.PReLU(),
                weight_norm(nn.Conv2d(20, classes, kernel_size=3, padding=1)),
                weight_norm(nn.Conv2d(classes, classes, kernel_size=3, padding=1))]
        sr_head = [weight_norm(nn.Conv2d(30, 48, kernel_size=3, padding=1)),
                   nn.PReLU(),
                   weight_norm(nn.Conv2d(48, 64, kernel_size=3, padding=1)),
                   nn.PReLU()]
        sr_body = [ResBlock(64, 6, 1.0, 0.75)
                   for _ in range(5)]
        sr_tail = [weight_norm(nn.Conv2d(64, 192, kernel_size=3, padding=1)),
                   nn.PReLU(),
                   weight_norm(nn.Conv2d(192, 192, kernel_size=3, padding=1)),
                   nn.PixelShuffle(8)]
        self.pre = Body_3()
        self.head = nn.Sequential(*head)
        self.body_1 = Body_1()
        self.body_2 = Body_2()
        self.tail = nn.Sequential(*tail)
        # self.skip = nn.Sequential(*skip)
        self.sr_head = nn.Sequential(*sr_head)
        self.sr_body = nn.Sequential(*sr_body)
        self.sr_tail = nn.Sequential(*sr_tail)

    def forward(self, x, y=None):
        img = x
        x = self.pre(x)
        img_sr = self.sr_head(x)
        img_sr = self.sr_body(img_sr)
        img_sr = self.sr_tail(img_sr)
        x = self.head(x)
        x1 = self.body_1(x)
        x = self.body_2(x1, x)
        # x = self.upsample(x)
        x = self.tail(x)
        if self.training:
            main_loss = self.criterion(x, y)
            aux_loss = 1 - ssim(img_sr, img)
            f = open('loss.txt', 'a')
            f.write(str(aux_loss))
            f.write('\n')
            f.close()
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

# from L-DETR & https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/det_pp_lcnet.py

# -*-coding:utf-8-*-
import numpy as np
import torch
# from util.misc import NestedTensor
import torch.nn as nn
# from .checkpoint import load_dygraph_pretrain

# from src.core import register
import sys

# __all__ = ['LCNet']


# k, in_c, out_c, s, use_se
NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}

MODEL_URLS = {
    0.25:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    0.35:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    0.5:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    0.75:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    1.0:
    # "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    # "https://github.com/wangjian123799/L-DETR/blob/main/torchpred/PPLCNet_x1_0_pretrained.pth.tar",
    "/content/LCNet/PPLCNet_x1_0_pretrained.pth.tar",
    # "C:\SHAFA\Ngoding\SKRIPSI-ngoding\RT-DETR\PPLCNet_x1_0_pretrained.pth.tar",
    1.5:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    2.0:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    2.5:
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
}


'''
   _make_divisible()作用将卷积核个数调整到8的整数倍。
    The function is to adjust the number of convolution kernels 
    to an integer multiple of 8.   
'''


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self, num_channels, filter_size, num_filters, stride, num_groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            filter_size,
            stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out


class DepthwiseSeparable(nn.Module):
    def __init__(self, num_channels, num_filters, stride, dw_size=3, use_se=False):
        super().__init__()
        self.use_se = use_se

        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)

        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


@register
class LCNet(nn.Module):
    def __init__(self, scale=1.0, class_num=1000, dropout_prob=0.2, class_expand=1280, pretrained=False):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand

        self.conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=make_divisible(16 * scale),
            stride=2,
            filter_size=3
        )
        # print("conv 1")
        # print(self.conv1.weight.shape)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])
        print("block 2")
        # print(self.blocks2)

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])
        print("block 3")
        # print(self.blocks3)

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])
        print("block 4")
        # print(self.blocks4)

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])
        print("block 5")
        # print(self.blocks5)

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])
        print("block 6")
        # print(self.blocks6)

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            out_channels=self. class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        print("last conv")
        # print(self.last_conv.weight.shape)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc = nn.Linear(self. class_expand, class_num)

        if pretrained:
            # state = torch.hub.load_state_dict_from_url(MODEL_URLS[scale])
            state = torch.load(MODEL_URLS[scale])
            self.load_state_dict(state)
            print(f'Load PPLCNet_x{scale} state_dict')

            # self._load_pretrained(
            #     MODEL_URLS["PPLCNet_x{}".format(scale)], use_ssld=use_ssld
            # )

        # sys.exit("~~~~~~~~~~~~~biar g error~~~~~~~~~~~~~~")

    def forward(self, x):

        x = self.conv1(x)
        x = self.blocks2(x)

        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        # x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        # x = self.dropout(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        return x


def PPLCNetEngine(scale=1.0, pretrained=None):
    model = LCNet(scale=scale, pretrained=pretrained)
    # if pretrained is not None:
    #     load_dygraph_pretrain(model, pretrained)
    return model

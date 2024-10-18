import torch
import torch.nn.functional as F
import torchvision

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchinfo import summary
from typing import Dict, List
from .lcnet import PPLCNetEngine

# def main()
'''
def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = PPLCNetEngine(
            scale=1.0, pretrained=r"/content/L-DETR/torchpred/PPLCNet_x1_0_ssld_pretrained")
        num_channels = 1280
        print(backbone)
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
'''

if __name__ == '__main__':
    print("halo")
    model = PPLCNetEngine(scale=1.0, pretrained=True)
    summary(model, input_size=(16, 1, 28, 28))

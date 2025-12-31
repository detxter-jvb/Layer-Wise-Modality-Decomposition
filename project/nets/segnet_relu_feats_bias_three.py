import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import pdb
import utils.geom
import utils.vox
import utils.misc
import utils.basic

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

from torch.autograd.functional import jacobian

EPS = 1e-4

from functools import partial

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x_to_upsample, cam_bev_):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([cam_bev_, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, cam_bev_, x_skip):
        return cam_bev_ + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()

        self.splitting_rule = "rule_2"
        self.norm_bias_rule =  "identity"
        self.norm_bias_rule_bn = "uniform"

        self.conservation_bias_split = True
        self.conservation_feat_sum = True
        self.conservation_act_sum = False

        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.activ_buffer = []
        self.norm_buffer = []

        self.layer1 = backbone.layer1
        self.layer1_1_conv1 = self.layer1[0].conv1
        self.layer1_1_bn1 = self.layer1[0].bn1

        self.layer1_1_relu = self.layer1[0].relu
        self.layer1_1_conv2 = self.layer1[0].conv2
        self.layer1_1_bn2 = self.layer1[0].bn2

        self.layer1_2_conv1 = self.layer1[1].conv1
        self.layer1_2_bn1 = self.layer1[1].bn1

        self.layer1_2_relu = self.layer1[1].relu
        self.layer1_2_conv2 = self.layer1[1].conv2
        self.layer1_2_bn2 = self.layer1[1].bn2

        self.layer2 = backbone.layer2
        self.layer2_1_conv1 = self.layer2[0].conv1
        self.layer2_1_bn1 = self.layer2[0].bn1

        self.layer2_1_relu = self.layer2[0].relu
        self.layer2_1_conv2 = self.layer2[0].conv2
        self.layer2_1_bn2 = self.layer2[0].bn2

        self.layer2_1_downsample_conv = self.layer2[0].downsample[0]
        self.layer2_1_downsample_bn = self.layer2[0].downsample[1]

        self.layer2_2_conv1 = self.layer2[1].conv1
        self.layer2_2_bn1 = self.layer2[1].bn1

        self.layer2_2_relu = self.layer2[1].relu
        self.layer2_2_conv2 = self.layer2[1].conv2
        self.layer2_2_bn2 = self.layer2[1].bn2

        self.layer3 = backbone.layer3
        self.layer3_1_conv1 = self.layer3[0].conv1
        self.layer3_1_bn1 = self.layer3[0].bn1

        self.layer3_1_relu = self.layer3[0].relu
        self.layer3_1_conv2 = self.layer3[0].conv2
        self.layer3_1_bn2 = self.layer3[0].bn2

        self.layer3_1_downsample_conv = self.layer3[0].downsample[0]
        self.layer3_1_downsample_bn = self.layer3[0].downsample[1]

        self.layer3_2_conv1 = self.layer3[1].conv1
        self.layer3_2_bn1 = self.layer3[1].bn1

        self.layer3_2_relu = self.layer3[1].relu
        self.layer3_2_conv2 = self.layer3[1].conv2
        self.layer3_2_bn2 = self.layer3[1].bn2

        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.up3_upsample = self.up3_skip.upsample_layer[0]
        self.up3_conv1 = self.up3_skip.upsample_layer[1]

        self.up2_upsample = self.up2_skip.upsample_layer[0]
        self.up2_conv1 = self.up2_skip.upsample_layer[1]

        self.up1_upsample = self.up1_skip.upsample_layer[0]
        self.up1_conv1 = self.up1_skip.upsample_layer[1]

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.seg_conv1 = self.segmentation_head[0]
        self.seg_instance_norm = self.segmentation_head[1]
        self.seg_act = self.segmentation_head[2]
        self.seg_conv2 = self.segmentation_head[3]

        self.instance_norm = nn.InstanceNorm2d(shared_out_channels)

        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def splitting_2(self, cam_feat, rad_feat, lid_feat, pseudo_feat):

        output_feat_1 = torch.zeros_like(rad_feat)
        output_feat_2 = torch.zeros_like(cam_feat)
        output_feat_3 = torch.zeros_like(lid_feat)
        output_feat_4 = torch.zeros_like(pseudo_feat)

        identity_condition = ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any()

        check = "identity"
        if check == "uniform" :
            output_feat_1 = torch.where(identity_condition, rad_feat + pseudo_feat/4, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat + pseudo_feat/4, output_feat_2)
            output_feat_3 = torch.where(identity_condition, lid_feat + pseudo_feat/4, output_feat_3)
            output_feat_4 = torch.where(identity_condition, pseudo_feat / 4, output_feat_4)
        elif check == "identity" :
            output_feat_1 = torch.where(identity_condition, rad_feat, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat, output_feat_2)
            output_feat_3 = torch.where(identity_condition, lid_feat, output_feat_3)
            output_feat_4 = torch.where(identity_condition, pseudo_feat, output_feat_4)

        return output_feat_1, output_feat_2, output_feat_3, output_feat_4

    def splitting(self, cam_feat, rad_feat, lid_feat, pseudo_feat):
        output_feat_1 = torch.zeros_like(rad_feat)
        output_feat_2 = torch.zeros_like(cam_feat)
        output_feat_3 = torch.zeros_like(lid_feat)
        output_feat_4 = torch.zeros_like(pseudo_feat)

        check = "sum"

        if check == "ratio" :

            identity_condition = ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                                 ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any()

            ratio_condition = ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                            ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                            ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                            ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any()

            ratio_condition_2 = ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                            ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                            ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                            ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any()

            ratio = (rad_feat) / (rad_feat + cam_feat)
            output_feat_1 = torch.where(ratio_condition, rad_feat + pseudo_feat * ratio, output_feat_1)
            output_feat_2 = torch.where(ratio_condition, cam_feat + pseudo_feat * (1-ratio), output_feat_2)
            output_feat_3 = torch.where(ratio_condition, torch.zeros_like(lid_feat), output_feat_3)
            output_feat_4 = torch.where(ratio_condition, torch.zeros_like(pseudo_feat), output_feat_4)

            output_feat_1 = torch.where(ratio_condition_2, rad_feat + pseudo_feat * (1 - ratio), output_feat_1)
            output_feat_2 = torch.where(ratio_condition_2, cam_feat + pseudo_feat * ratio, output_feat_2)
            output_feat_3 = torch.where(ratio_condition_2, torch.zeros_like(lid_feat), output_feat_3)
            output_feat_4 = torch.where(ratio_condition_2, torch.zeros_like(pseudo_feat), output_feat_4)

            output_feat_1 = torch.where(identity_condition, rad_feat, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat, output_feat_2)
            output_feat_3 = torch.where(identity_condition, lid_feat, output_feat_3)
            output_feat_4 = torch.where(identity_condition, pseudo_feat, output_feat_4)

        elif check == "sum":

            identity_condition = ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                                ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                                ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                                ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any()

            sum_condition_cam = ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat < 0) & (rad_feat >= 0) & (lid_feat > 0) & (pseudo_feat < 0)).any() or \
                                ((cam_feat < 0) & (rad_feat >= 0) & (lid_feat <= 0) & (pseudo_feat < 0)).any()
            sum_condition_rad = ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat >= 0) & (rad_feat < 0) & (lid_feat > 0) & (pseudo_feat < 0)).any() or \
                                ((cam_feat >= 0) & (rad_feat < 0) & (lid_feat <= 0) & (pseudo_feat < 0)).any()

            output_feat_1 = torch.where(identity_condition, rad_feat, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat, output_feat_2)
            output_feat_3 = torch.where(identity_condition, lid_feat, output_feat_3)
            output_feat_4 = torch.where(identity_condition, pseudo_feat, output_feat_4)

            output_feat_1 = torch.where(sum_condition_rad, rad_feat + pseudo_feat, output_feat_1)
            output_feat_2 = torch.where(sum_condition_rad, cam_feat, output_feat_2)
            output_feat_3 = torch.where(sum_condition_rad, torch.zeros_like(lid_feat), output_feat_3)
            output_feat_4 = torch.where(sum_condition_rad, torch.zeros_like(pseudo_feat), output_feat_4)

            output_feat_1 = torch.where(sum_condition_cam, rad_feat, output_feat_1)
            output_feat_2 = torch.where(sum_condition_cam, cam_feat + pseudo_feat, output_feat_2)
            output_feat_3 = torch.where(sum_condition_cam, torch.zeros_like(lid_feat), output_feat_3)
            output_feat_4 = torch.where(sum_condition_cam, torch.zeros_like(pseudo_feat), output_feat_4)

        return output_feat_1, output_feat_2, output_feat_3, output_feat_4

    def forward(self, feat_bev_, cam_bev_, rad_bev_, lid_bev_, pseudo_bev_, bev_flip_indices=None, act_mask=None, norm_mask=None):
        b, c, h, w = cam_bev_.shape

        self.bn1.eval()
        self.layer1_1_bn1.eval()
        self.layer1_1_bn2.eval()
        self.layer1_2_bn1.eval()
        self.layer1_2_bn2.eval()
        self.layer2_1_bn1.eval()
        self.layer2_1_bn2.eval()
        self.layer2_1_downsample_bn.eval()
        self.layer2_2_bn1.eval()
        self.layer2_2_bn2.eval()
        self.layer3_1_bn1.eval()
        self.layer3_1_bn2.eval()
        self.layer3_1_downsample_bn.eval()
        self.layer3_2_bn1.eval()
        self.layer3_2_bn2.eval()

        skip_feat = {'1': feat_bev_}
        skip_cam = {'1': cam_bev_}
        skip_rad = {'1': rad_bev_}
        skip_lid = {'1' : lid_bev_}
        skip_psd = {'1': pseudo_bev_}

        # import pdb; pdb.set_trace()

        feat_bev_ = self.first_conv(feat_bev_)
        cam_bev_ = self.first_conv(cam_bev_)
        rad_bev_ = self.first_conv(rad_bev_)
        lid_bev_ = self.first_conv(lid_bev_)
        pseudo_bev_ = self.first_conv(pseudo_bev_)

        temp_bev_ = feat_bev_.clone()
        feat_bev_ = self.bn1(feat_bev_)

        if self.norm_bias_rule_bn == "uniform" :
            feat_bev_2 = (temp_bev_ - self.bn1.running_mean[None, :, None, None]) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            feat_bev_2 = feat_bev_2 * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None]

            cam_bev_ = (cam_bev_ - self.bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            cam_bev_ = cam_bev_ * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None] / 4

            lid_bev_ = (lid_bev_ - self.bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            lid_bev_ = lid_bev_ * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None] / 4

            rad_bev_ = (rad_bev_ - self.bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            rad_bev_ = rad_bev_ * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None] / 4

            pseudo_bev_ = (pseudo_bev_ - self.bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            pseudo_bev_ = pseudo_bev_ * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio" :

            feat_bev_2 = (temp_bev_ - self.bn1.running_mean[None, :, None, None]) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            feat_bev_2 = feat_bev_2 * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None]

            cam_bev_ = (cam_bev_) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            cam_bev_ = cam_bev_ * self.bn1.weight[None, :, None, None]

            lid_bev_ = (lid_bev_) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            lid_bev_ = lid_bev_ * self.bn1.weight[None, :, None, None]

            rad_bev_ = (rad_bev_) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            rad_bev_ = rad_bev_ * self.bn1.weight[None, :, None, None]

            pseudo_bev_ = (pseudo_bev_ - self.bn1.running_mean[None, :, None, None]) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            pseudo_bev_ = pseudo_bev_ * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity" :
            feat_bev_2 = (temp_bev_ - self.bn1.running_mean[None, :, None, None]) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            feat_bev_2 = feat_bev_2 * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None]

            cam_bev_ = (cam_bev_) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            cam_bev_ = cam_bev_ * self.bn1.weight[None, :, None, None]

            lid_bev_ = (lid_bev_) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            lid_bev_ = lid_bev_ * self.bn1.weight[None, :, None, None]

            rad_bev_ = (rad_bev_) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            rad_bev_ = rad_bev_ * self.bn1.weight[None, :, None, None]

            pseudo_bev_ = (pseudo_bev_ - self.bn1.running_mean[None, :, None, None]) / torch.sqrt(self.bn1.running_var[None, :, None, None] + self.bn1.eps)
            pseudo_bev_ = pseudo_bev_ * self.bn1.weight[None, :, None, None] + self.bn1.bias[None, :, None, None]

        if self.splitting_rule == "rule_1" :
            rad_bev_, cam_bev_, lid_bev_, pseudo_bev_ = self.splitting(cam_bev_, rad_bev_, lid_bev_, pseudo_bev_)

        elif self.splitting_rule == "rule_2" :
            rad_bev_, cam_bev_, lid_bev_, pseudo_bev_ = self.splitting_2(cam_bev_, rad_bev_, lid_bev_, pseudo_bev_)

        if act_mask is not None :
            feat_bev_ = act_mask['act_feat_2'] * feat_bev_
            cam_bev_ = act_mask['act_feat_2'] * cam_bev_
            rad_bev_ = act_mask['act_feat_2'] * rad_bev_
            lid_bev_ = act_mask['act_feat_2'] * lid_bev_
            pseudo_bev_ = act_mask['act_feat_2'] * pseudo_bev_

        else:
            cam_bev_ = self.relu(cam_bev_)
            self.activ_buffer.append(cam_bev_ > 0)

        identity_feat = feat_bev_
        identity_cam = cam_bev_
        identity_rad = rad_bev_
        identity_lid = lid_bev_
        identity_psd = pseudo_bev_

        out_feat = self.layer1_1_conv1(feat_bev_)
        out_cam = self.layer1_1_conv1(cam_bev_)
        out_rad = self.layer1_1_conv1(rad_bev_)
        out_lid = self.layer1_1_conv1(lid_bev_)
        out_psd = self.layer1_1_conv1(pseudo_bev_)

        if self.norm_bias_rule_bn == "uniform" :

            out_feat = self.layer1_1_bn1(out_feat)

            out_cam = (out_cam - self.layer1_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_cam = out_cam * self.layer1_1_bn1.weight[None, :, None, None] + self.layer1_1_bn1.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer1_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_rad = out_rad * self.layer1_1_bn1.weight[None, :, None, None] + self.layer1_1_bn1.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer1_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_lid = out_lid * self.layer1_1_bn1.weight[None, :, None, None] + self.layer1_1_bn1.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer1_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_psd = out_psd * self.layer1_1_bn1.weight[None, :, None, None] + self.layer1_1_bn1.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio":

            out_feat = self.layer1_1_bn1(out_feat)
            out_cam = (out_cam) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_cam = out_cam * self.layer1_1_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_rad = out_rad * self.layer1_1_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_lid = out_lid * self.layer1_1_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_1_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_psd = out_psd * self.layer1_1_bn1.weight[None, :, None, None] + self.layer1_1_bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity" :
            out_feat = self.layer1_1_bn1(out_feat)
            out_cam = (out_cam) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_cam = out_cam * self.layer1_1_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_rad = out_rad * self.layer1_1_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_lid = out_lid * self.layer1_1_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_1_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_1_bn1.running_var[None, :, None, None] + self.layer1_1_bn1.eps)
            out_psd = out_psd * self.layer1_1_bn1.weight[None, :, None, None] + self.layer1_1_bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "grad" :
            out_feat = self.layer1_1_bn1(out_feat)
            print("lets start !")

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2":
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            out_feat = act_mask['act_feat_3'] * out_feat
            out_cam = act_mask['act_feat_3'] * out_cam
            out_rad = act_mask['act_feat_3'] * out_rad
            out_lid = act_mask['act_feat_3'] * out_lid
            out_psd = act_mask['act_feat_3'] * out_psd
        else:
            out_cam = self.layer1_1_relu(out_cam)
            self.activ_buffer.append(out_cam > 0)

        out_feat = self.layer1_1_conv2(out_feat)
        out_cam = self.layer1_1_conv2(out_cam)
        out_rad = self.layer1_1_conv2(out_rad)
        out_lid = self.layer1_1_conv2(out_lid)
        out_psd = self.layer1_1_conv2(out_psd)

        out_feat = self.layer1_1_bn2(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer1_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_cam = out_cam * self.layer1_1_bn2.weight[None, :, None, None] + self.layer1_1_bn2.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer1_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_rad = out_rad * self.layer1_1_bn2.weight[None, :, None, None] + self.layer1_1_bn2.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer1_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_lid = out_lid * self.layer1_1_bn2.weight[None, :, None, None] + self.layer1_1_bn2.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer1_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_psd = out_psd * self.layer1_1_bn2.weight[None, :, None, None] + self.layer1_1_bn2.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_cam = out_cam * self.layer1_1_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_rad = out_rad * self.layer1_1_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_lid = out_lid * self.layer1_1_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_1_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_psd = out_psd * self.layer1_1_bn2.weight[None, :, None, None] + self.layer1_1_bn2.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_cam = out_cam * self.layer1_1_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_rad = out_rad * self.layer1_1_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_lid = out_lid * self.layer1_1_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_1_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_1_bn2.running_var[None, :, None, None] + self.layer1_1_bn2.eps)
            out_psd = out_psd * self.layer1_1_bn2.weight[None, :, None, None] + self.layer1_1_bn2.bias[None, :, None, None]

        out_feat += identity_feat
        out_cam += identity_cam
        out_rad += identity_rad
        out_lid += identity_lid
        out_psd += identity_psd

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            feat_bev_ = act_mask['act_feat_4'] * out_feat
            cam_bev_ = act_mask['act_feat_4'] * out_cam
            rad_bev_ = act_mask['act_feat_4'] * out_rad
            lid_bev_ = act_mask['act_feat_4'] * out_lid
            psd_bev_ = act_mask['act_feat_4'] * out_psd
        else:
            feat_bev_ = self.relu(out_feat)
            cam_bev_ = self.relu(out_cam)
            rad_bev_ = self.relu(out_rad)
            lid_bev_ = self.relu(out_lid)
            psd_bev_ = self.relu(out_psd)

        identity_feat = feat_bev_
        identity_cam = cam_bev_
        identity_rad = rad_bev_
        identity_lid = lid_bev_
        identity_psd = psd_bev_

        out_feat = self.layer1_2_conv1(feat_bev_)
        out_cam = self.layer1_2_conv1(cam_bev_)
        out_rad = self.layer1_2_conv1(rad_bev_)
        out_lid = self.layer1_2_conv1(lid_bev_)
        out_psd = self.layer1_2_conv1(psd_bev_)

        out_feat = self.layer1_2_bn1(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer1_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_cam = out_cam * self.layer1_2_bn1.weight[None, :, None, None] + self.layer1_2_bn1.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer1_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_rad = out_rad * self.layer1_2_bn1.weight[None, :, None, None] + self.layer1_2_bn1.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer1_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_lid = out_lid * self.layer1_2_bn1.weight[None, :, None, None] + self.layer1_2_bn1.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer1_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_psd = out_psd * self.layer1_2_bn1.weight[None, :, None, None] + self.layer1_2_bn1.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_cam = out_cam * self.layer1_2_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_lid = out_lid * self.layer1_2_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_rad = out_rad * self.layer1_2_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_2_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_psd = out_psd * self.layer1_2_bn1.weight[None, :, None, None] + self.layer1_2_bn1.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_cam = out_cam * self.layer1_2_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_lid = out_lid * self.layer1_2_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_rad = out_rad * self.layer1_2_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_2_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_2_bn1.running_var[None, :, None, None] + self.layer1_2_bn1.eps)
            out_psd = out_psd * self.layer1_2_bn1.weight[None, :, None, None] + self.layer1_2_bn1.bias[None, :, None, None]

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            out_feat = act_mask['act_feat_5'] * out_feat
            out_cam = act_mask['act_feat_5'] * out_cam
            out_rad = act_mask['act_feat_5'] * out_rad
            out_lid = act_mask['act_feat_5'] * out_lid
            out_psd = act_mask['act_feat_5'] * out_psd
        else:
            out_cam = self.layer1_2_relu(out_cam)
            self.activ_buffer.append(out_cam > 0)

        out_feat = self.layer1_2_conv2(out_feat)
        out_cam = self.layer1_2_conv2(out_cam)
        out_rad = self.layer1_2_conv2(out_rad)
        out_lid = self.layer1_2_conv2(out_lid)
        out_psd = self.layer1_2_conv2(out_psd)

        out_feat = self.layer1_2_bn2(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer1_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_cam = out_cam * self.layer1_2_bn2.weight[None, :, None, None] + self.layer1_2_bn2.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer1_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_lid = out_lid * self.layer1_2_bn2.weight[None, :, None, None] + self.layer1_2_bn2.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer1_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_rad = out_rad * self.layer1_2_bn2.weight[None, :, None, None] + self.layer1_2_bn2.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer1_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_psd = out_psd * self.layer1_2_bn2.weight[None, :, None, None] + self.layer1_2_bn2.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_cam = out_cam * self.layer1_2_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_lid = out_lid * self.layer1_2_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_rad = out_rad * self.layer1_2_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_2_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_psd = out_psd * self.layer1_2_bn2.weight[None, :, None, None] + self.layer1_2_bn2.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_cam = out_cam * self.layer1_2_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_lid = out_lid * self.layer1_2_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_rad = out_rad * self.layer1_2_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer1_2_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer1_2_bn2.running_var[None, :, None, None] + self.layer1_2_bn2.eps)
            out_psd = out_psd * self.layer1_2_bn2.weight[None, :, None, None] + self.layer1_2_bn2.bias[None, :, None, None]

        out_feat += identity_feat
        out_cam += identity_cam
        out_rad += identity_rad
        out_lid += identity_lid
        out_psd += identity_psd

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            feat_bev_ = act_mask['act_feat_6'] * out_feat
            cam_bev_ = act_mask['act_feat_6'] * out_cam
            rad_bev_ = act_mask['act_feat_6'] * out_rad
            lid_bev_ = act_mask['act_feat_6'] * out_lid
            psd_bev_ = act_mask['act_feat_6'] * out_psd
        else:
            cam_bev_ = self.layer1_2_relu(out_cam)
            lid_bev_ = self.layer1_2_relu(out_lid)
            self.activ_buffer.append(cam_bev_ > 0)

        skip_feat['2'] = feat_bev_
        skip_cam['2'] = cam_bev_
        skip_rad['2'] = rad_bev_
        skip_lid['2'] = lid_bev_
        skip_psd['2'] = psd_bev_

        identity_feat = feat_bev_
        identity_cam = cam_bev_
        identity_rad = rad_bev_
        identity_lid = lid_bev_
        identity_psd = psd_bev_

        out_feat = self.layer2_1_conv1(feat_bev_)
        out_cam = self.layer2_1_conv1(cam_bev_)
        out_rad = self.layer2_1_conv1(rad_bev_)
        out_lid = self.layer2_1_conv1(lid_bev_)
        out_psd = self.layer2_1_conv1(psd_bev_)

        out_feat = self.layer2_1_bn1(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer2_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_cam = out_cam * self.layer2_1_bn1.weight[None, :, None, None] + self.layer2_1_bn1.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer2_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_lid = out_lid * self.layer2_1_bn1.weight[None, :, None, None] + self.layer2_1_bn1.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer2_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_rad = out_rad * self.layer2_1_bn1.weight[None, :, None, None] + self.layer2_1_bn1.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer2_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_psd = out_psd * self.layer2_1_bn1.weight[None, :, None, None] + self.layer2_1_bn1.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_cam = out_cam * self.layer2_1_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_lid = out_lid * self.layer2_1_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_rad = out_rad * self.layer2_1_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_1_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_psd = out_psd * self.layer2_1_bn1.weight[None, :, None, None] + self.layer2_1_bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_cam = out_cam * self.layer2_1_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_lid = out_lid * self.layer2_1_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_rad = out_rad * self.layer2_1_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_1_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_1_bn1.running_var[None, :, None, None] + self.layer2_1_bn1.eps)
            out_psd = out_psd * self.layer2_1_bn1.weight[None, :, None, None] + self.layer2_1_bn1.bias[None, :, None, None]

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            out_feat = act_mask['act_feat_7'] * out_feat
            out_cam = act_mask['act_feat_7'] * out_cam
            out_rad = act_mask['act_feat_7'] * out_rad
            out_lid = act_mask['act_feat_7'] * out_lid
            out_psd = act_mask['act_feat_7'] * out_psd
        else:
            out_cam = self.layer2_1_relu(out_cam)
            self.activ_buffer.append(out_cam > 0)

        out_feat = self.layer2_1_conv2(out_feat)
        out_cam = self.layer2_1_conv2(out_cam)
        out_rad = self.layer2_1_conv2(out_rad)
        out_lid = self.layer2_1_conv2(out_lid)
        out_psd = self.layer2_1_conv2(out_psd)

        out_feat = self.layer2_1_bn2(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer2_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_cam = out_cam * self.layer2_1_bn2.weight[None, :, None, None] + self.layer2_1_bn2.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer2_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_lid = out_lid * self.layer2_1_bn2.weight[None, :, None, None] + self.layer2_1_bn2.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer2_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_rad = out_rad * self.layer2_1_bn2.weight[None, :, None, None] + self.layer2_1_bn2.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer2_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_psd = out_psd * self.layer2_1_bn2.weight[None, :, None, None] + self.layer2_1_bn2.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_cam = out_cam * self.layer2_1_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_lid = out_lid * self.layer2_1_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_rad = out_rad * self.layer2_1_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_1_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_psd = out_psd * self.layer2_1_bn2.weight[None, :, None, None] + self.layer2_1_bn2.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_cam = out_cam * self.layer2_1_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_rad = out_rad * self.layer2_1_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_1_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_1_bn2.running_var[None, :, None, None] + self.layer2_1_bn2.eps)
            out_psd = out_psd * self.layer2_1_bn2.weight[None, :, None, None] + self.layer2_1_bn2.bias[None, :, None, None]

        identity_feat = self.layer2_1_downsample_conv(feat_bev_)
        identity_cam = self.layer2_1_downsample_conv(cam_bev_)
        identity_lid = self.layer2_1_downsample_conv(lid_bev_)
        identity_rad = self.layer2_1_downsample_conv(rad_bev_)
        identity_psd = self.layer2_1_downsample_conv(psd_bev_)

        identity_feat = self.layer2_1_downsample_bn(identity_feat)

        if self.norm_bias_rule_bn == "uniform" :

            identity_cam = (identity_cam - self.layer2_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_cam = identity_cam * self.layer2_1_downsample_bn.weight[None, :, None, None] + self.layer2_1_downsample_bn.bias[None, :, None, None] / 4

            identity_lid = (identity_lid - self.layer2_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_lid = identity_lid * self.layer2_1_downsample_bn.weight[None, :, None, None] + self.layer2_1_downsample_bn.bias[None, :, None, None] / 4

            identity_rad = (identity_rad - self.layer2_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_rad = identity_rad * self.layer2_1_downsample_bn.weight[None, :, None, None] + self.layer2_1_downsample_bn.bias[None, :, None, None] / 4

            identity_psd = (identity_psd - self.layer2_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_psd = identity_psd * self.layer2_1_downsample_bn.weight[None, :, None, None] + self.layer2_1_downsample_bn.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            identity_cam = (identity_cam) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_cam = identity_cam * self.layer2_1_downsample_bn.weight[None, :, None, None]

            identity_lid = (identity_lid) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_lid = identity_lid * self.layer2_1_downsample_bn.weight[None, :, None, None]

            identity_rad = (identity_rad) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_rad = identity_rad * self.layer2_1_downsample_bn.weight[None, :, None, None]

            identity_psd = (identity_psd - self.layer2_1_downsample_bn.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_psd = identity_psd * self.layer2_1_downsample_bn.weight[None, :, None, None] + self.layer2_1_downsample_bn.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :

            identity_cam = (identity_cam) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_cam = identity_cam * self.layer2_1_downsample_bn.weight[None, :, None, None]

            identity_lid = (identity_lid) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_lid = identity_lid * self.layer2_1_downsample_bn.weight[None, :, None, None]

            identity_rad = (identity_rad) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_rad = identity_rad * self.layer2_1_downsample_bn.weight[None, :, None, None]

            identity_psd = (identity_psd - self.layer2_1_downsample_bn.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_1_downsample_bn.running_var[None, :, None, None] + self.layer2_1_downsample_bn.eps)
            identity_psd = identity_psd * self.layer2_1_downsample_bn.weight[None, :, None, None] + self.layer2_1_downsample_bn.bias[None, :, None, None]

        out_feat += identity_feat
        out_cam += identity_cam
        out_rad += identity_rad
        out_lid += identity_lid
        out_psd += identity_psd

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            feat_bev_ = act_mask['act_feat_8'] * out_feat
            cam_bev_ = act_mask['act_feat_8'] * out_cam
            lid_bev_ = act_mask['act_feat_8'] * out_lid
            rad_bev_ = act_mask['act_feat_8'] * out_rad
            psd_bev_ = act_mask['act_feat_8'] * out_psd
        else:
            cam_bev_ = self.layer2_1_relu(out_cam)
            feat_bev_ = self.layer2_1_relu(out_feat)
            lid_bev_ = self.layer2_1_relu(out_lid)
            rad_bev_ = self.layer2_1_relu(out_rad)
            psd_bev_ = self.layer2_1_relu(out_psd)
            self.activ_buffer.append(cam_bev_ > 0)

        identity_feat = feat_bev_
        identity_cam = cam_bev_
        identity_lid = lid_bev_
        identity_rad = rad_bev_
        identity_psd = psd_bev_

        out_feat = self.layer2_2_conv1(feat_bev_)
        out_cam = self.layer2_2_conv1(cam_bev_)
        out_lid = self.layer2_2_conv1(lid_bev_)
        out_rad = self.layer2_2_conv1(rad_bev_)
        out_psd = self.layer2_2_conv1(psd_bev_)

        out_feat = self.layer2_2_bn1(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer2_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_cam = out_cam * self.layer2_2_bn1.weight[None, :, None, None] + self.layer2_2_bn1.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer2_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_lid = out_lid * self.layer2_2_bn1.weight[None, :, None, None] + self.layer2_2_bn1.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer2_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_rad = out_rad * self.layer2_2_bn1.weight[None, :, None, None] + self.layer2_2_bn1.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer2_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_psd = out_psd * self.layer2_2_bn1.weight[None, :, None, None] + self.layer2_2_bn1.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_cam = out_cam * self.layer2_2_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_lid = out_lid * self.layer2_2_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_rad = out_rad * self.layer2_2_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_2_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_psd = out_psd * self.layer2_2_bn1.weight[None, :, None, None] + self.layer2_2_bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_cam = out_cam * self.layer2_2_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_lid = out_lid * self.layer2_2_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_rad = out_rad * self.layer2_2_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_2_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_2_bn1.running_var[None, :, None, None] + self.layer2_2_bn1.eps)
            out_psd = out_psd * self.layer2_2_bn1.weight[None, :, None, None] + self.layer2_2_bn1.bias[None, :, None, None]

        if self.splitting_rule == "rule_1":
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2":
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            out_feat = act_mask['act_feat_9'] * out_feat
            out_cam = act_mask['act_feat_9'] * out_cam
            out_lid = act_mask['act_feat_9'] * out_lid
            out_rad = act_mask['act_feat_9'] * out_rad
            out_psd = act_mask['act_feat_9'] * out_psd
        else:
            out_cam = self.layer2_2_relu(out_cam)
            self.activ_buffer.append(out_cam > 0)

        out_feat = self.layer2_2_conv2(out_feat)
        out_cam = self.layer2_2_conv2(out_cam)
        out_lid = self.layer2_2_conv2(out_lid)
        out_rad = self.layer2_2_conv2(out_rad)
        out_psd = self.layer2_2_conv2(out_psd)

        out_feat = self.layer2_2_bn2(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer2_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_cam = out_cam * self.layer2_2_bn2.weight[None, :, None, None] + self.layer2_2_bn2.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer2_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_lid = out_lid * self.layer2_2_bn2.weight[None, :, None, None] + self.layer2_2_bn2.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer2_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_rad = out_rad * self.layer2_2_bn2.weight[None, :, None, None] + self.layer2_2_bn2.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer2_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_psd = out_psd * self.layer2_2_bn2.weight[None, :, None, None] + self.layer2_2_bn2.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_cam = out_cam * self.layer2_2_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_lid = out_lid * self.layer2_2_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_rad = out_rad * self.layer2_2_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_2_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_psd = out_psd * self.layer2_2_bn2.weight[None, :, None, None] + self.layer2_2_bn2.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_cam = out_cam * self.layer2_2_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_lid = out_lid * self.layer2_2_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_rad = out_rad * self.layer2_2_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer2_2_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer2_2_bn2.running_var[None, :, None, None] + self.layer2_2_bn2.eps)
            out_psd = out_psd * self.layer2_2_bn2.weight[None, :, None, None] + self.layer2_2_bn2.bias[None, :, None, None]

        out_feat += identity_feat
        out_cam += identity_cam
        out_lid += identity_lid
        out_rad += identity_rad
        out_psd += identity_psd

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            feat_bev_ = act_mask['act_feat_10'] * out_feat
            cam_bev_ = act_mask['act_feat_10'] * out_cam
            lid_bev_ = act_mask['act_feat_10'] * out_lid
            rad_bev_ = act_mask['act_feat_10'] * out_rad
            psd_bev_ = act_mask['act_feat_10'] * out_psd
        else:
            cam_bev_ = self.layer2_2_relu(out_cam)
            feat_bev_ = self.layer2_2_relu(out_feat)
            lid_bev_ = self.layer2_2_relu(out_lid)
            rad_bev_ = self.layer2_2_relu(out_rad)
            psd_bev_ = self.layer2_2_relu(out_psd)
            self.activ_buffer.append(cam_bev_ > 0)

        skip_feat['3'] = feat_bev_
        skip_cam['3'] = cam_bev_
        skip_lid['3'] = lid_bev_
        skip_rad['3'] = rad_bev_
        skip_psd['3'] = psd_bev_

        identity_feat = feat_bev_
        identity_cam = cam_bev_
        identity_lid = lid_bev_
        identity_rad = rad_bev_
        identity_psd = psd_bev_

        out_feat = self.layer3_1_conv1(feat_bev_)
        out_cam = self.layer3_1_conv1(cam_bev_)
        out_lid = self.layer3_1_conv1(lid_bev_)
        out_rad = self.layer3_1_conv1(rad_bev_)
        out_psd = self.layer3_1_conv1(psd_bev_)

        out_feat = self.layer3_1_bn1(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer3_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_cam = out_cam * self.layer3_1_bn1.weight[None, :, None, None] + self.layer3_1_bn1.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer3_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_lid = out_lid * self.layer3_1_bn1.weight[None, :, None, None] + self.layer3_1_bn1.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer3_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_rad = out_rad * self.layer3_1_bn1.weight[None, :, None, None] + self.layer3_1_bn1.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer3_1_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_psd = out_psd * self.layer3_1_bn1.weight[None, :, None, None] + self.layer3_1_bn1.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_cam = out_cam * self.layer3_1_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_lid = out_lid * self.layer3_1_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_rad = out_rad * self.layer3_1_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_1_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_psd = out_psd * self.layer3_1_bn1.weight[None, :, None, None] + self.layer3_1_bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_cam = out_cam * self.layer3_1_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_lid = out_lid * self.layer3_1_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_rad = out_rad * self.layer3_1_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_1_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_1_bn1.running_var[None, :, None, None] + self.layer3_1_bn1.eps)
            out_psd = out_psd * self.layer3_1_bn1.weight[None, :, None, None] + self.layer3_1_bn1.bias[None, :, None, None]

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            out_feat = act_mask['act_feat_11'] * out_feat
            out_cam = act_mask['act_feat_11'] * out_cam
            out_lid = act_mask['act_feat_11'] * out_lid
            out_rad = act_mask['act_feat_11'] * out_rad
            out_psd = act_mask['act_feat_11'] * out_psd
        else:
            out_cam = self.layer3_1_relu(out_cam)
            self.activ_buffer.append(out_cam > 0)

        out_feat = self.layer3_1_conv2(out_feat)
        out_cam = self.layer3_1_conv2(out_cam)
        out_lid = self.layer3_1_conv2(out_lid)
        out_rad = self.layer3_1_conv2(out_rad)
        out_psd = self.layer3_1_conv2(out_psd)

        out_feat = self.layer3_1_bn2(out_feat)

        if self.norm_bias_rule_bn == "uniform":

            out_cam = (out_cam - self.layer3_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_cam = out_cam * self.layer3_1_bn2.weight[None, :, None, None] + self.layer3_1_bn2.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer3_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_lid = out_lid * self.layer3_1_bn2.weight[None, :, None, None] + self.layer3_1_bn2.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer3_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_rad = out_rad * self.layer3_1_bn2.weight[None, :, None, None] + self.layer3_1_bn2.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer3_1_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_psd = out_psd * self.layer3_1_bn2.weight[None, :, None, None] + self.layer3_1_bn2.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio":

            out_cam = (out_cam) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_cam = out_cam * self.layer3_1_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_lid = out_lid * self.layer3_1_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_rad = out_rad * self.layer3_1_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_1_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_psd = out_psd * self.layer3_1_bn2.weight[None, :, None, None] + self.layer3_1_bn2.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity":

            out_cam = (out_cam) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_cam = out_cam * self.layer3_1_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_lid = out_lid * self.layer3_1_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_rad = out_rad * self.layer3_1_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_1_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_1_bn2.running_var[None, :, None, None] + self.layer3_1_bn2.eps)
            out_psd = out_psd * self.layer3_1_bn2.weight[None, :, None, None] + self.layer3_1_bn2.bias[None, :, None, None]

        identity_feat = self.layer3_1_downsample_conv(feat_bev_)
        identity_cam = self.layer3_1_downsample_conv(cam_bev_)
        identity_lid = self.layer3_1_downsample_conv(lid_bev_)
        identity_rad = self.layer3_1_downsample_conv(rad_bev_)
        identity_psd = self.layer3_1_downsample_conv(psd_bev_)

        identity_feat = self.layer3_1_downsample_bn(identity_feat)

        if self.norm_bias_rule_bn == "uniform" :
            identity_cam = (identity_cam - self.layer3_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_cam = identity_cam * self.layer3_1_downsample_bn.weight[None, :, None, None] + self.layer3_1_downsample_bn.bias[None, :, None, None] / 4

            identity_lid = (identity_lid - self.layer3_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_lid = identity_lid * self.layer3_1_downsample_bn.weight[None, :, None, None] + self.layer3_1_downsample_bn.bias[None, :, None, None] / 4

            identity_rad = (identity_rad - self.layer3_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_rad = identity_rad * self.layer3_1_downsample_bn.weight[None, :, None, None] + self.layer3_1_downsample_bn.bias[None, :, None, None] / 4

            identity_psd = (identity_psd - self.layer3_1_downsample_bn.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_psd = identity_psd * self.layer3_1_downsample_bn.weight[None, :, None, None] + self.layer3_1_downsample_bn.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            identity_cam = (identity_cam) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_cam = identity_cam * self.layer3_1_downsample_bn.weight[None, :, None, None]

            identity_lid = (identity_lid) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_lid = identity_lid * self.layer3_1_downsample_bn.weight[None, :, None, None]

            identity_rad = (identity_rad) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_rad = identity_rad * self.layer3_1_downsample_bn.weight[None, :, None, None]

            identity_psd = (identity_psd - self.layer3_1_downsample_bn.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_psd = identity_psd * self.layer3_1_downsample_bn.weight[None, :, None, None] + self.layer3_1_downsample_bn.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :
            identity_cam = (identity_cam) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_cam = identity_cam * self.layer3_1_downsample_bn.weight[None, :, None, None]

            identity_lid = (identity_lid) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_lid = identity_lid * self.layer3_1_downsample_bn.weight[None, :, None, None]

            identity_rad = (identity_rad) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_rad = identity_rad * self.layer3_1_downsample_bn.weight[None, :, None, None]

            identity_psd = (identity_psd - self.layer3_1_downsample_bn.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_1_downsample_bn.running_var[None, :, None, None] + self.layer3_1_downsample_bn.eps)
            identity_psd = identity_psd * self.layer3_1_downsample_bn.weight[None, :, None, None] + self.layer3_1_downsample_bn.bias[None, :, None, None]

        out_feat += identity_feat
        out_cam += identity_cam
        out_lid += identity_lid
        out_rad += identity_rad
        out_psd += identity_psd

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            feat_bev_ = act_mask['act_feat_12'] * out_feat
            cam_bev_ = act_mask['act_feat_12'] * out_cam
            lid_bev_ = act_mask['act_feat_12'] * out_lid
            rad_bev_ = act_mask['act_feat_12'] * out_rad
            psd_bev_ = act_mask['act_feat_12'] * out_psd
        else:
            cam_bev_ = self.layer3_1_relu(out_cam)
            feat_bev_ = self.layer3_1_relu(out_feat)
            lid_bev_ = self.layer3_1_relu(out_lid)
            rad_bev_ = self.layer3_1_relu(out_rad)
            psd_bev_ = self.layer3_1_relu(out_psd)
            self.activ_buffer.append(cam_bev_ > 0)

        identity_feat = feat_bev_
        identity_cam = cam_bev_
        identity_lid = lid_bev_
        identity_rad = rad_bev_
        identity_psd = psd_bev_

        out_feat = self.layer3_2_conv1(feat_bev_)
        out_cam = self.layer3_2_conv1(cam_bev_)
        out_lid = self.layer3_2_conv1(lid_bev_)
        out_rad = self.layer3_2_conv1(rad_bev_)
        out_psd = self.layer3_2_conv1(psd_bev_)

        out_feat = self.layer3_2_bn1(out_feat)

        if self.norm_bias_rule_bn == "uniform":

            out_cam = (out_cam - self.layer3_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_cam = out_cam * self.layer3_2_bn1.weight[None, :, None, None] + self.layer3_2_bn1.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer3_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_lid = out_lid * self.layer3_2_bn1.weight[None, :, None, None] + self.layer3_2_bn1.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer3_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_rad = out_rad * self.layer3_2_bn1.weight[None, :, None, None] + self.layer3_2_bn1.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer3_2_bn1.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_psd = out_psd * self.layer3_2_bn1.weight[None, :, None, None] + self.layer3_2_bn1.bias[None, :, None, None] / 4

        elif self.norm_bias_rule_bn == "ratio":

            out_cam = (out_cam) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_cam = out_cam * self.layer3_2_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_lid = out_lid * self.layer3_2_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_rad = out_rad * self.layer3_2_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_2_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_psd = out_psd * self.layer3_2_bn1.weight[None, :, None, None] + self.layer3_2_bn1.bias[None, :, None, None]

        elif self.norm_bias_rule_bn == "identity":

            out_cam = (out_cam) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_cam = out_cam * self.layer3_2_bn1.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_lid = out_lid * self.layer3_2_bn1.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_rad = out_rad * self.layer3_2_bn1.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_2_bn1.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_2_bn1.running_var[None, :, None, None] + self.layer3_2_bn1.eps)
            out_psd = out_psd * self.layer3_2_bn1.weight[None, :, None, None] + self.layer3_2_bn1.bias[None, :, None, None]

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            out_feat = act_mask['act_feat_13'] * out_feat
            out_cam = act_mask['act_feat_13'] * out_cam
            out_lid = act_mask['act_feat_13'] * out_lid
            out_rad = act_mask['act_feat_13'] * out_rad
            out_psd = act_mask['act_feat_13'] * out_psd
        else:
            out_cam = self.layer3_2_relu(out_cam)
            self.activ_buffer.append(out_cam > 0)

        out_feat = self.layer3_2_conv2(out_feat)
        out_cam = self.layer3_2_conv2(out_cam)
        out_lid = self.layer3_2_conv2(out_lid)
        out_rad = self.layer3_2_conv2(out_rad)
        out_psd = self.layer3_2_conv2(out_psd)

        out_feat = self.layer3_2_bn2(out_feat)

        if self.norm_bias_rule_bn == "uniform" :

            out_cam = (out_cam - self.layer3_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_cam = out_cam * self.layer3_2_bn2.weight[None, :, None, None] + self.layer3_2_bn2.bias[None, :, None, None] / 4

            out_lid = (out_lid - self.layer3_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_lid = out_lid * self.layer3_2_bn2.weight[None, :, None, None] + self.layer3_2_bn2.bias[None, :, None, None] / 4

            out_rad = (out_rad - self.layer3_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_rad = out_rad * self.layer3_2_bn2.weight[None, :, None, None] + self.layer3_2_bn2.bias[None, :, None, None] / 4

            out_psd = (out_psd - self.layer3_2_bn2.running_mean[None, :, None, None] / 4) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_psd = out_psd * self.layer3_2_bn2.weight[None, :, None, None] + self.layer3_2_bn2.bias[None, :, None, None] / 4

        if self.norm_bias_rule_bn == "ratio" :

            out_cam = (out_cam) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_cam = out_cam * self.layer3_2_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_lid = out_lid * self.layer3_2_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_rad = out_rad * self.layer3_2_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_2_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_psd = out_psd * self.layer3_2_bn2.weight[None, :, None, None] + self.layer3_2_bn2.bias[None, :, None, None]

        if self.norm_bias_rule_bn == "identity" :

            out_cam = (out_cam) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_cam = out_cam * self.layer3_2_bn2.weight[None, :, None, None]

            out_lid = (out_lid) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_lid = out_lid * self.layer3_2_bn2.weight[None, :, None, None]

            out_rad = (out_rad) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_rad = out_rad * self.layer3_2_bn2.weight[None, :, None, None]

            out_psd = (out_psd - self.layer3_2_bn2.running_mean[None, :, None, None]) / torch.sqrt(self.layer3_2_bn2.running_var[None, :, None, None] + self.layer3_2_bn2.eps)
            out_psd = out_psd * self.layer3_2_bn2.weight[None, :, None, None] + self.layer3_2_bn2.bias[None, :, None, None]

        out_feat += identity_feat
        out_cam += identity_cam
        out_lid += identity_lid
        out_rad += identity_rad
        out_psd += identity_psd

        if self.splitting_rule == "rule_1" :
            out_rad, out_cam, out_lid, out_psd  = self.splitting(out_cam, out_rad, out_lid, out_psd)
        elif self.splitting_rule == "rule_2" :
            out_rad, out_cam, out_lid, out_psd = self.splitting_2(out_cam, out_rad, out_lid, out_psd)

        if act_mask is not None:
            feat_bev_ = act_mask['act_feat_14'] * out_feat
            cam_bev_ = act_mask['act_feat_14'] * out_cam
            rad_bev_ = act_mask['act_feat_14'] * out_rad
            lid_bev_ = act_mask['act_feat_14'] * out_lid
            psd_bev_ = act_mask['act_feat_14'] * out_psd
        else:
            cam_bev_ = self.layer3_2_relu(out_cam)
            self.activ_buffer.append(cam_bev_ > 0)

        feat_bev_ = self.up3_upsample(feat_bev_)
        cam_bev_ = self.up3_upsample(cam_bev_)
        rad_bev_ = self.up3_upsample(rad_bev_)
        lid_bev_ = self.up3_upsample(lid_bev_)
        psd_bev_ = self.up3_upsample(psd_bev_)

        feat_bev_ = self.up3_conv1(feat_bev_)
        cam_bev_ = self.up3_conv1(cam_bev_)
        rad_bev_ = self.up3_conv1(rad_bev_)
        lid_bev_ = self.up3_conv1(lid_bev_)
        psd_bev_ = self.up3_conv1(psd_bev_)

        if self.splitting_rule == "rule_1" :
            rad_bev_, cam_bev_, lid_bev_, psd_bev_ = self.splitting(cam_bev_, rad_bev_, lid_bev_, psd_bev_)
        elif self.splitting_rule == "rule_2" :
            rad_bev_, cam_bev_, lid_bev_, psd_bev_ = self.splitting_2(cam_bev_, rad_bev_, lid_bev_, psd_bev_)

        if norm_mask is not None :
            if self.norm_bias_rule == "uniform":

                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                cam_bev_ = (cam_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                lid_bev_ = (lid_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) / 4) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                rad_bev_ = (rad_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                psd_bev_ = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))

            elif self.norm_bias_rule == "ratio":
                cam_mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
                rad_mean = torch.mean(rad_bev_, dim=(2,3), keepdim=True)
                lid_mean = torch.mean(lid_bev_, dim=(2,3), keepdim=True)
                pseudo_mean = torch.mean(psd_bev_, dim=(2,3), keepdim=True)

                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                cam_bev_ = (cam_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                rad_bev_ = (rad_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                lid_bev_ = (lid_bev_ - lid_mean) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                psd_bev_ = (psd_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))

            elif self.norm_bias_rule == "identity" :
                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                cam_bev_ = (cam_bev_) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                rad_bev_ = (rad_bev_) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                lid_bev_ = (lid_bev_) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))
                psd_bev_ = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-7))

        else :
            mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
            var = torch.var(cam_bev_, dim=(2,3), unbiased=False, keepdim=True)
            cam_bev_ = (cam_bev_ - mean) / torch.sqrt(var+1e-5)
            self.norm_buffer.append(mean)
            self.norm_buffer.append(var)

        feat_bev_ = self.up3_skip(feat_bev_, skip_feat['3'])
        cam_bev_ = self.up3_skip(cam_bev_, skip_cam['3'])
        rad_bev_ = self.up3_skip(rad_bev_, skip_rad['3'])
        lid_bev_ = self.up3_skip(lid_bev_, skip_lid['3'])
        psd_bev_ = self.up3_skip(psd_bev_, skip_psd['3'])

        up3_activations_feat = feat_bev_
        up3_activations_cam = cam_bev_
        up3_activations_rad = rad_bev_
        up3_activations_lid = lid_bev_
        up3_activations_psd = psd_bev_

        feat_bev_ = self.up2_upsample(feat_bev_)
        cam_bev_ = self.up2_upsample(cam_bev_)
        rad_bev_ = self.up2_upsample(rad_bev_)
        lid_bev_ = self.up2_upsample(lid_bev_)
        psd_bev_ = self.up2_upsample(psd_bev_)

        feat_bev_ = self.up2_conv1(feat_bev_)
        cam_bev_ = self.up2_conv1(cam_bev_)
        rad_bev_ = self.up2_conv1(rad_bev_)
        lid_bev_ = self.up2_conv1(lid_bev_)
        psd_bev_ = self.up2_conv1(psd_bev_)

        if norm_mask is not None :
            if self.norm_bias_rule == "uniform":

                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                cam_bev_ = (cam_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                rad_bev_ = (rad_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                lid_bev_ = (lid_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                psd_bev_ = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))

            elif self.norm_bias_rule == "ratio":
                cam_mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
                rad_mean = torch.mean(rad_bev_, dim=(2,3), keepdim=True)
                lid_mean = torch.mean(lid_bev_, dim=(2,3), keepdim=True)
                pseudo_mean = torch.mean(psd_bev_, dim=(2,3), keepdim=True)

                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                cam_bev_ = (cam_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                rad_bev_ = (rad_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                lid_bev_ = (lid_bev_ - lid_mean) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                psd_bev_ = (psd_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))

            elif self.norm_bias_rule == "identity" :
                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                cam_bev_ = (cam_bev_) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                rad_bev_ = (rad_bev_) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                lid_bev_ = (lid_bev_) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))
                psd_bev_ = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-7))

        else :
            mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
            var = torch.var(cam_bev_, dim=(2,3), unbiased=False, keepdim=True)
            cam_bev_ = (cam_bev_ - mean) / torch.sqrt(var+1e-5)
            self.norm_buffer.append(mean)
            self.norm_buffer.append(var)

        feat_bev_ = self.up2_skip(feat_bev_, skip_feat['2'])
        cam_bev_ = self.up2_skip(cam_bev_, skip_cam['2'])
        rad_bev_ = self.up2_skip(rad_bev_, skip_rad['2'])
        lid_bev_ = self.up2_skip(lid_bev_, skip_lid['2'])
        psd_bev_ = self.up2_skip(psd_bev_, skip_psd['2'])

        up2_activations_feat = feat_bev_
        up2_activations_cam = cam_bev_
        up2_activations_rad = rad_bev_
        up2_activations_lid = lid_bev_
        up2_activations_psd = psd_bev_

        feat_bev_ = self.up1_upsample(feat_bev_)
        cam_bev_ = self.up1_upsample(cam_bev_)
        rad_bev_ = self.up1_upsample(rad_bev_)
        lid_bev_ = self.up1_upsample(lid_bev_)
        psd_bev_ = self.up1_upsample(psd_bev_)

        feat_bev_ = self.up1_conv1(feat_bev_)
        cam_bev_ = self.up1_conv1(cam_bev_)
        rad_bev_ = self.up1_conv1(rad_bev_)
        lid_bev_ = self.up1_conv1(lid_bev_)
        psd_bev_ = self.up1_conv1(psd_bev_)

        if self.splitting_rule == "rule_1" :
            rad_bev_, cam_bev_, lid_bev_, psd_bev_ = self.splitting(cam_bev_, rad_bev_, lid_bev_, psd_bev_)
        elif self.splitting_rule == "rule_2" :
            rad_bev_, cam_bev_, lid_bev_, psd_bev_ = self.splitting_2(cam_bev_, rad_bev_, lid_bev_, psd_bev_)

        if norm_mask is not None :
            if self.norm_bias_rule == "uniform" :

                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                cam_bev_ = (cam_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                rad_bev_ = (rad_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                lid_bev_ = (lid_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                psd_bev_ = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))

            elif self.norm_bias_rule == "ratio":
                cam_mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
                rad_mean = torch.mean(rad_bev_, dim=(2,3), keepdim=True)
                lid_mean = torch.mean(lid_bev_, dim=(2,3), keepdim=True)
                pseudo_mean = torch.mean(psd_bev_, dim=(2,3), keepdim=True)

                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                cam_bev_ = (cam_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                rad_bev_ = (rad_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                lid_bev_ = (lid_bev_ - lid_mean) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                psd_bev_ = (psd_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))

            elif self.norm_bias_rule == "identity" :
                feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                cam_bev_ = (cam_bev_) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                rad_bev_ = (rad_bev_) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                lid_bev_ = (lid_bev_) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))
                psd_bev_ = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-7))

        else :
            mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
            var = torch.var(cam_bev_, dim=(2,3), unbiased=False, keepdim=True)
            cam_bev_ = (cam_bev_ - mean) / torch.sqrt(var+1e-5)
            self.norm_buffer.append(mean)
            self.norm_buffer.append(var)

        feat_bev_ = self.up1_skip(feat_bev_, skip_feat['1'])
        cam_bev_ = self.up1_skip(cam_bev_, skip_cam['1'])
        rad_bev_ = self.up1_skip(rad_bev_, skip_rad['1'])
        lid_bev_ = self.up1_skip(lid_bev_, skip_lid['1'])
        psd_bev_ = self.up1_skip(psd_bev_, skip_psd['1'])

        up1_activations_feat = feat_bev_
        up1_activations_cam = cam_bev_
        up1_activations_rad = rad_bev_
        up1_activations_lid = lid_bev_
        up1_activations_psd = psd_bev_

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            cam_bev_[bev_flip2_index] = torch.flip(cam_bev_[bev_flip2_index], [-2])
            cam_bev_[bev_flip1_index] = torch.flip(cam_bev_[bev_flip1_index], [-1])

        feat_output_feat = self.feat_head(feat_bev_)
        feat_output_cam = self.feat_head(cam_bev_)
        feat_output_rad = self.feat_head(rad_bev_)
        feat_output_lid = self.feat_head(lid_bev_)
        feat_output_psd = self.feat_head(psd_bev_)

        feat_bev_ = self.seg_conv1(feat_bev_)
        cam_bev_ = self.seg_conv1(cam_bev_)
        lid_bev_ = self.seg_conv1(lid_bev_)
        rad_bev_ = self.seg_conv1(rad_bev_)
        psd_bev_ = self.seg_conv1(psd_bev_)

        if self.splitting_rule == "rule_1" :
            rad_bev_, cam_bev_, lid_bev_, psd_bev_ = self.splitting(cam_bev_, rad_bev_, lid_bev_, psd_bev_)
        if self.splitting_rule == "rule_2" :
            rad_bev_, cam_bev_, lid_bev_, psd_bev_ = self.splitting_2(cam_bev_, rad_bev_, lid_bev_, psd_bev_)

        if norm_mask is not None :
            if self.norm_bias_rule == "uniform" :
                seg_in_feat = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_cam = (cam_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) / 4) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_lid = (lid_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) / 4) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_rad = (rad_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) / 4) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_psd = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) / 4) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)

            elif self.norm_bias_rule == "ratio":
                cam_mean = torch.mean(cam_bev_, dim=(2,3), keepdim=True)
                lid_mean = torch.mean(lid_bev_, dim=(2,3), keepdim=True)
                rad_mean = torch.mean(rad_bev_, dim=(2,3), keepdim=True)
                pseudo_mean = torch.mean(psd_bev_, dim=(2,3), keepdim=True)

                seg_in_feat = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_5_var']+1e-7))
                seg_in_cam = (cam_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_5_var']+1e-7))
                seg_in_lid = (lid_bev_ - lid_mean) / torch.sqrt((norm_mask['norm_feat_5_var']+1e-7))
                seg_in_rad = (rad_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_5_var']+1e-7))
                seg_in_psd = (psd_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_5_var']+1e-7))

            elif self.norm_bias_rule == "identity" :
                seg_in_feat = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_cam = (cam_bev_) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_lid = (lid_bev_) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_rad = (rad_bev_) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)
                seg_in_psd = (psd_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-7)

        else:
            # When norm_mask is None, process all modalities
            mean_feat = torch.mean(feat_bev_, dim=(2,3), keepdim=True)
            var_feat = torch.var(feat_bev_, dim=(2,3), unbiased=False, keepdim=True)
            self.norm_buffer.append(mean_feat)
            self.norm_buffer.append(var_feat)
            
            seg_in_feat = (feat_bev_ - mean_feat) / torch.sqrt(var_feat+1e-7)
            seg_in_cam = (cam_bev_ - mean_feat) / torch.sqrt(var_feat+1e-7)
            seg_in_rad = (rad_bev_ - mean_feat) / torch.sqrt(var_feat+1e-7)
            seg_in_lid = (lid_bev_ - mean_feat) / torch.sqrt(var_feat+1e-7)
            seg_in_psd = (psd_bev_ - mean_feat) / torch.sqrt(var_feat+1e-7)

        if self.splitting_rule == "rule_1" :
            seg_in_rad, seg_in_cam, seg_in_lid, seg_in_psd = self.splitting(seg_in_cam, seg_in_rad, seg_in_lid, seg_in_psd)
        if self.splitting_rule == "rule_2" :
            seg_in_rad, seg_in_cam, seg_in_lid, seg_in_psd = self.splitting_2(seg_in_cam, seg_in_rad, seg_in_lid, seg_in_psd)

        if act_mask is not None:
            seg_out_feat = act_mask['act_feat_15'] * seg_in_feat
            seg_out_cam = act_mask['act_feat_15'] * seg_in_cam
            seg_out_lid = act_mask['act_feat_15'] * seg_in_lid
            seg_out_rad = act_mask['act_feat_15'] * seg_in_rad
            seg_out_psd = act_mask['act_feat_15'] * seg_in_psd
        else:
            seg_out_feat = self.seg_act(seg_in_feat)
            self.activ_buffer.append(seg_out_cam > 0)

        final_bias = torch.zeros_like(seg_out_psd)
        segmentation_output_feat = self.seg_conv2(seg_out_feat)
        segmentation_output_cam = self.seg_conv2(seg_out_cam)
        segmentation_output_lid = self.seg_conv2(seg_out_lid)
        segmentation_output_rad = self.seg_conv2(seg_out_rad)
        segmentation_output_psd = self.seg_conv2(seg_out_psd)
        final_bias = self.seg_conv2(final_bias)

        instance_center_output_feat = self.instance_center_head(feat_bev_)
        instance_center_output_cam = self.instance_center_head(cam_bev_)
        instance_center_output_lid = self.instance_center_head(lid_bev_)
        instance_center_output_rad = self.instance_center_head(rad_bev_)
        instance_center_output_psd = self.instance_center_head(psd_bev_)

        instance_offset_output_feat = self.instance_offset_head(feat_bev_)
        instance_offset_output_cam = self.instance_offset_head(cam_bev_)
        instance_offset_output_lid = self.instance_offset_head(lid_bev_)
        instance_offset_output_rad = self.instance_offset_head(rad_bev_)
        instance_offset_output_psd = self.instance_offset_head(psd_bev_)

        instance_future_output = self.instance_future_head(cam_bev_) if self.predict_future_flow else None

        return {
            'act_feat': self.activ_buffer,
            'norm_feat' : self.norm_buffer,
            'up3_activations_feat' : up3_activations_feat,
            'up3_activations_cam' : up3_activations_cam,
            'up3_activations_rad' : up3_activations_rad,
            'up3_activations_lid' : up3_activations_lid,
            'up3_activations_psd' : up3_activations_psd,
            'up2_activations_feat' : up2_activations_feat,
            'up2_activations_cam' : up2_activations_cam,
            'up2_activations_rad' : up2_activations_rad,
            'up2_activations_lid' : up2_activations_lid,
            'up2_activations_psd' : up2_activations_psd,
            'up1_activations_feat': up1_activations_feat,
            'up1_activations_cam' : up1_activations_cam,
            'up1_activations_rad' : up1_activations_rad,
            'up1_activations_lid' : up1_activations_lid,
            'up1_activations_psd' : up1_activations_psd,
            'feat': feat_output_cam.view(b, *feat_output_cam.shape[1:]),
            'segmentation_feat' : segmentation_output_feat.view(b, *segmentation_output_feat.shape[1:]),
            'segmentation_cam': segmentation_output_cam.view(b, *segmentation_output_cam.shape[1:]),
            'segmentation_lid': segmentation_output_lid.view(b, *segmentation_output_lid.shape[1:]),
            'segmentation_rad': segmentation_output_rad.view(b, *segmentation_output_rad.shape[1:]),
            'segmentation_psd': segmentation_output_psd.view(b, *segmentation_output_psd.shape[1:]),

            'segmentation_feat_sigmoid' : torch.sigmoid(segmentation_output_feat).view(b, *segmentation_output_feat.shape[1:]),
            'segmentation_cam_sigmoid' : torch.sigmoid(segmentation_output_cam).view(b, *segmentation_output_cam.shape[1:]),
            'segmentation_lid_sigmoid' : torch.sigmoid(segmentation_output_lid).view(b, *segmentation_output_lid.shape[1:]),
            'segmentation_rad_sigmoid' : torch.sigmoid(segmentation_output_rad).view(b, *segmentation_output_rad.shape[1:]),
            'segmentation_psd_sigmoid' : torch.sigmoid(segmentation_output_psd).view(b, *segmentation_output_psd.shape[1:]),

            'segmentation_feat_sigmoid_round' : torch.sigmoid(segmentation_output_feat).view(b, *segmentation_output_feat.shape[1:]).round(),
            'segmentation_cam_sigmoid_round' : torch.sigmoid(segmentation_output_cam).view(b, *segmentation_output_cam.shape[1:]).round(),
            'segmentation_lid_sigmoid_round' : torch.sigmoid(segmentation_output_lid).view(b, *segmentation_output_lid.shape[1:]).round(),
            'segmentation_rad_sigmoid_round' : torch.sigmoid(segmentation_output_rad).view(b, *segmentation_output_rad.shape[1:]).round(),
            'segmentation_psd_sigmoid_round' : torch.sigmoid(segmentation_output_psd).view(b, *segmentation_output_psd.shape[1:]).round(),

            'instance_center_feat': instance_center_output_feat.view(b, *instance_center_output_feat.shape[1:]),
            'instance_center_cam': instance_center_output_cam.view(b, *instance_center_output_cam.shape[1:]),
            'instance_center_lid': instance_center_output_lid.view(b, *instance_center_output_lid.shape[1:]),
            'instance_center_rad': instance_center_output_rad.view(b, *instance_center_output_rad.shape[1:]),
            'instance_center_psd': instance_center_output_psd.view(b, *instance_center_output_psd.shape[1:]),
            'instance_offset_feat': instance_offset_output_feat.view(b, *instance_offset_output_feat.shape[1:]),
            'instance_offset_cam': instance_offset_output_cam.view(b, *instance_offset_output_cam.shape[1:]),
            'instance_offset_lid': instance_offset_output_lid.view(b, *instance_offset_output_lid.shape[1:]),
            'instance_offset_rad': instance_offset_output_rad.view(b, *instance_offset_output_rad.shape[1:]),
            'instance_offset_psd': instance_offset_output_psd.view(b, *instance_offset_output_psd.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import torchvision

class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, cam_bev_):
        x1 = self.backbone(cam_bev_)
        x2 = self.layer3(x1)
        cam_bev_ = self.upsampling_layer(x2, x1)
        cam_bev_ = self.depth_layer(cam_bev_)

        return cam_bev_

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, cam_bev_):
        x1 = self.backbone(cam_bev_)
        x2 = self.layer3(x1)
        cam_bev_ = self.upsampling_layer(x2, x1)
        cam_bev_ = self.depth_layer(cam_bev_)

        return cam_bev_

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, cam_bev_):

        endpoints = dict()

        cam_bev_ = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(cam_bev_)))
        prev_x = cam_bev_

        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            cam_bev_ = block(cam_bev_, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > cam_bev_.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = cam_bev_

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        endpoints['reduction_{}'.format(len(endpoints) + 1)] = cam_bev_

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']

        cam_bev_ = self.upsampling_layer(input_1, input_2)
        return cam_bev_

    def forward(self, cam_bev_):
        cam_bev_ = self.get_features(cam_bev_)
        cam_bev_ = self.depth_layer(cam_bev_)
        return cam_bev_

class Segnet_relu_feats_bias(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None,
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 act_mask=None,
                 norm_mask=None,
                 encoder_type="res101"):
        super(Segnet_relu_feats_bias, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.splitting_rule = "rule_2"
        self.norm_bias_rule =  "ratio"
        self.norm_bias_rule_bn = "identity"

        self.conservation_bias_split = True
        self.conservation_feat_sum = True
        self.conservation_act_sum = False

        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()

        self.activation = nn.GELU()
        self.act_mask = act_mask
        norm_mask = norm_mask

        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:

            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        if self.use_radar and self.use_lidar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y +Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                )
        elif self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim)
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),

            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:

                pass

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

    def splitting_2(self, cam_feat, rad_feat, lid_feat, pseudo_feat):

        output_feat_1 = torch.zeros_like(rad_feat)
        output_feat_2 = torch.zeros_like(cam_feat)
        output_feat_3 = torch.zeros_like(lid_feat)
        output_feat_4 = torch.zeros_like(pseudo_feat)

        identity_condition = ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (lid_feat <= 0) & (pseudo_feat > 0)).any()

        check = "identity"
        if check == "uniform" :
            output_feat_1 = torch.where(identity_condition, rad_feat + pseudo_feat/4, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat + pseudo_feat/4, output_feat_2)
            output_feat_3 = torch.where(identity_condition, lid_feat + pseudo_feat/4, output_feat_3)
            output_feat_4 = torch.where(identity_condition, pseudo_feat / 4, output_feat_4)
        elif check == "identity" :
            output_feat_1 = torch.where(identity_condition, rad_feat, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat, output_feat_2)
            output_feat_3 = torch.where(identity_condition, lid_feat, output_feat_3)
            output_feat_4 = torch.where(identity_condition, pseudo_feat, output_feat_4)

        return output_feat_1, output_feat_2, output_feat_3, output_feat_4

    def splitting(self, cam_feat, rad_feat, pseudo_feat):

        output_feat_1 = torch.zeros_like(rad_feat)
        output_feat_2 = torch.zeros_like(cam_feat)
        output_feat_3 = torch.zeros_like(pseudo_feat)

        check = "sum"

        if check == "ratio" :

            identity_condition = ((cam_feat > 0) & (rad_feat <= 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat > 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat > 0) & (rad_feat <= 0) & (pseudo_feat <= 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat > 0) & (pseudo_feat <= 0)).any()

            ratio_condition = ((cam_feat > 0) & (rad_feat > 0) & (pseudo_feat > 0)).any() or \
                            ((cam_feat <= 0) & (rad_feat <= 0) & (pseudo_feat <= 0)).any()

            ratio_condition_2 = ((cam_feat > 0) & (rad_feat > 0) & (pseudo_feat <= 0)).any() or \
                            ((cam_feat <= 0) & (rad_feat <= 0) & (pseudo_feat > 0))

            ratio = (rad_feat) / (rad_feat + cam_feat)
            output_feat_1 = torch.where(ratio_condition, rad_feat + pseudo_feat * ratio, output_feat_1)
            output_feat_2 = torch.where(ratio_condition, cam_feat + pseudo_feat * (1-ratio), output_feat_2)
            output_feat_3 = torch.where(ratio_condition, torch.zeros_like(pseudo_feat), output_feat_3)

            output_feat_1 = torch.where(ratio_condition_2, rad_feat + pseudo_feat * (1 - ratio), output_feat_1)
            output_feat_2 = torch.where(ratio_condition_2, cam_feat + pseudo_feat * ratio, output_feat_2)
            output_feat_3 = torch.where(ratio_condition_2, torch.zeros_like(pseudo_feat), output_feat_3)

            output_feat_1 = torch.where(identity_condition, rad_feat, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat, output_feat_2)
            output_feat_3 = torch.where(identity_condition, pseudo_feat, output_feat_3)

        elif check == "sum":

            identity_condition = ((cam_feat > 0) & (rad_feat > 0) & (pseudo_feat <= 0)).any() or \
                                ((cam_feat <= 0) & (rad_feat <= 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat > 0) & (rad_feat > 0) & (pseudo_feat > 0)).any() or \
                                 ((cam_feat <= 0) & (rad_feat <= 0) & (pseudo_feat <= 0)).any()

            sum_condition_cam = ((cam_feat > 0) & (rad_feat <= 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat < 0) & (rad_feat >= 0) & (pseudo_feat < 0)).any()
            sum_condition_rad = ((cam_feat <= 0) & (rad_feat > 0) & (pseudo_feat > 0)).any() or \
                                ((cam_feat >= 0) & (rad_feat < 0) & (pseudo_feat < 0)).any()

            output_feat_1 = torch.where(identity_condition, rad_feat, output_feat_1)
            output_feat_2 = torch.where(identity_condition, cam_feat, output_feat_2)
            output_feat_3 = torch.where(identity_condition, pseudo_feat, output_feat_3)

            output_feat_1 = torch.where(sum_condition_rad, rad_feat + pseudo_feat, output_feat_1)
            output_feat_2 = torch.where(sum_condition_rad, cam_feat, output_feat_2)
            output_feat_3 = torch.where(sum_condition_rad, torch.zeros_like(pseudo_feat), output_feat_3)

            output_feat_1 = torch.where(sum_condition_cam, rad_feat, output_feat_1)
            output_feat_2 = torch.where(sum_condition_cam, cam_feat + pseudo_feat, output_feat_2)
            output_feat_3 = torch.where(sum_condition_cam, torch.zeros_like(pseudo_feat), output_feat_3)

        return output_feat_1, output_feat_2, output_feat_3, output_feat_4

    def splitting_3(self, cam_feat, rad_feat, pseudo_feat):

        output_feat_1 = torch.zeros_like(rad_feat)
        output_feat_2 = torch.zeros_like(cam_feat)
        output_feat_3 = torch.zeros_like(pseudo_feat)

        ratio_condition = (cam_feat > 0) & (rad_feat > 0) & (pseudo_feat > 0)

        identity_condition = ((cam_feat > 0) & (rad_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat <= 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat <=0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (pseudo_feat > 0)).any() or \
                             ((cam_feat <= 0) & (rad_feat > 0) & (pseudo_feat <= 0)).any() or \
                             ((cam_feat > 0) & (rad_feat > 0) & (pseudo_feat > 0)).any()

        output_feat_1 = torch.where(identity_condition, rad_feat + pseudo_feat / 3, output_feat_1)
        output_feat_2 = torch.where(identity_condition, cam_feat + pseudo_feat / 3, output_feat_2)
        output_feat_3 = torch.where(identity_condition, pseudo_feat / 3, output_feat_3)

        return output_feat_1, output_feat_2, output_feat_3, output_feat_4

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None, act_mask=None, norm_mask=None):
        self.act_mask = act_mask
        norm_mask = norm_mask

        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)

        __p = lambda cam_bev_: utils.basic.pack_seqdim(cam_bev_, B)
        __u = lambda cam_bev_: utils.basic.unpack_seqdim(cam_bev_, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)

        if self.rand_flip:
            print("should not be inside here")
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_)

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        norm_e = dict()

        if self.use_radar and self.use_lidar:
            camera_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            lid_bev_ = rad_occ_mem0[0].permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            rad_bev_ = rad_occ_mem0[1].permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
            feat_bev_ = torch.cat([camera_bev_, rad_bev_, lid_bev_], dim=1)

            pseudo_camera = torch.zeros_like(camera_bev_)
            pseudo_lidar = torch.zeros_like(lid_bev_)
            pseudo_radar = torch.zeros_like(rad_bev_)

            # 각 모달리티별 feature를 준비 (다른 모달리티 자리는 pseudo로 채움)
            camera_bev_ = torch.cat([camera_bev_, pseudo_radar, pseudo_lidar], dim=1)
            radar_bev_ = torch.cat([pseudo_camera, rad_bev_, pseudo_lidar], dim=1)
            lidar_bev_ = torch.cat([pseudo_camera, pseudo_radar, lid_bev_], dim=1)
            pseudo_bev_ = torch.cat([pseudo_camera, pseudo_radar, pseudo_lidar], dim=1)

            # BEV compression
            feat_bev_ = self.bev_compressor(feat_bev_)
            camera_bev_ = self.bev_compressor(camera_bev_)
            radar_bev_ = self.bev_compressor(radar_bev_)
            lidar_bev_ = self.bev_compressor(lidar_bev_)
            pseudo_bev_ = self.bev_compressor(pseudo_bev_)

            # print("check for debug")
            # import pdb; pdb.set_trace()
        
            # Pseudo feature 제거
            camera_bev_ = camera_bev_ - pseudo_bev_
            radar_bev_ = radar_bev_ - pseudo_bev_
            lidar_bev_ = lidar_bev_ - pseudo_bev_

            if self.conservation_feat_sum :
                if not torch.allclose(feat_bev_, camera_bev_ + radar_bev_ + lidar_bev_ + pseudo_bev_, atol=1e-02, rtol=1e-05):
                    pdb.set_trace()

            if norm_mask is not None:
                if self.norm_bias_rule == "uniform":
                    feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    radar_bev_ = (radar_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    camera_bev_ = (camera_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    lidar_bev_ = (lidar_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    pseudo_bev_ = (pseudo_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /4) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                elif self.norm_bias_rule == "ratio":
                    cam_mean = torch.mean(camera_bev_, dim=(2,3), keepdim=True)
                    rad_mean = torch.mean(radar_bev_, dim=(2,3), keepdim=True)
                    lid_mean = torch.mean(lidar_bev_, dim=(2,3), keepdim=True)
                    pseudo_mean = torch.mean(pseudo_bev_, dim=(2,3), keepdim=True)

                    feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    radar_bev_ = (radar_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    camera_bev_ = (camera_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    lidar_bev_ = (lidar_bev_ - lid_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    pseudo_bev_ = (pseudo_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                elif self.norm_bias_rule == "identity" :
                    feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    radar_bev_ = (radar_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    camera_bev_ = (camera_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    lidar_bev_ = (lidar_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    pseudo_bev_ = (pseudo_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

            if self.act_mask is not None:
                feat_bev_ = feat_bev_ * self.act_mask['act_feat_1']
                camera_bev_ = camera_bev_ * self.act_mask['act_feat_1']
                radar_bev_ = radar_bev_ * self.act_mask['act_feat_1']
                lidar_bev_ = lidar_bev_ * self.act_mask['act_feat_1']
                pseudo_bev_ = pseudo_bev_ * self.act_mask['act_feat_1']
            else:
                feat_bev = self.activation(feat_bev_)

            out_dict = self.decoder(feat_bev_, camera_bev_, radar_bev_, lidar_bev_, pseudo_bev_, 
                                   (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None, 
                                   self.act_mask, norm_mask)

        elif self.use_radar:
            assert(rad_occ_mem0 is not None)
            if not self.use_metaradar:
                print("should not be inside here")
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1)
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev_ = self.bev_compressor(feat_bev_)
                if self.act_mask is not None:
                    feat_bev = feat_bev_ * self.act_mask['act_feat_1']
                else:
                    feat_bev = self.activation(feat_bev_)
            else:
                camera_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)

                pseudo_camera = torch.zeros_like(camera_bev_)
                pseudo_radar = torch.zeros_like(rad_bev_)

                feat_bev_ = torch.cat([camera_bev_, rad_bev_], dim=1)
                camera_bev_ = torch.cat([camera_bev_, pseudo_radar], dim=1)
                radar_bev_ = torch.cat([pseudo_camera, rad_bev_], dim=1)
                pseudo_bev_ = torch.cat([pseudo_camera, pseudo_radar], dim=1)

                feat_bev_ = self.bev_compressor(feat_bev_)
                radar_bev_ = self.bev_compressor(radar_bev_)
                camera_bev_ = self.bev_compressor(camera_bev_)
                pseudo_bev_ = self.bev_compressor(pseudo_bev_)

                radar_bev_ = radar_bev_ - pseudo_bev_
                camera_bev_ = camera_bev_ - pseudo_bev_

                if self.conservation_feat_sum :
                    if not torch.allclose(feat_bev_, camera_bev_ + radar_bev_ + pseudo_bev_, atol=1e-02, rtol=1e-05):
                        pdb.set_trace()

                if norm_mask is not None:

                    if self.norm_bias_rule == "uniform":

                        feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        radar_bev_ = (radar_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /3) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        camera_bev_ = (camera_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /3) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        pseudo_bev_ = (pseudo_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /3) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                    elif self.norm_bias_rule == "ratio":
                        cam_mean = torch.mean(camera_bev_, dim=(2,3), keepdim=True)
                        rad_mean = torch.mean(radar_bev_, dim=(2,3), keepdim=True)
                        pseudo_mean = torch.mean(pseudo_bev_, dim=(2,3), keepdim=True)

                        feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        radar_bev_ = (radar_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        camera_bev_ = (camera_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        pseudo_bev_ = (pseudo_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                    elif self.norm_bias_rule == "identity" :
                        feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        radar_bev_ = (radar_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        camera_bev_ = (camera_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                        pseudo_bev_ = (pseudo_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                    elif self.norm_bias_rule == "grad" :

                        camera_bev_ = camera_bev_.detach().clone().requires_grad_()
                        radar_bev_ = radar_bev_.detach().clone().requires_grad_()
                        pseudo_bev_ = pseudo_bev_.detach().clone().requires_grad_()

                        feat_bev_ = camera_bev_ + radar_bev_ + pseudo_bev_
                        feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt(torch.var(feat_bev_, dim=(2,3), unbiased=False, keepdim=True)+1e-7)

                        camera_bev_grad = torch.autograd.functional.jacobian(feat_bev_, camera_bev_)
                        radar_bev_grad = torch.autograd.functional.jacobian(feat_bev_, radar_bev_)

                        camera_bev_ = torch.mul(camera_bev_grad, camera_bev_)
                        radar_bev_ = torch.mul(radar_bev_grad, radar_bev_)
                        pseudo_bev_ = feat_bev_ - (camera_bev_ + radar_bev_)

                else:
                    mean = torch.mean(feat_bev_, dim=(2,3), keepdim=True)
                    var = torch.var(feat_bev_, dim=(2,3), unbiased=False, keepdim=True)
                    radar_bev_ = (feat_bev_- mean) / torch.sqrt((var+1e-5))
                    norm_e['norm_feat_1_mean'] = mean
                    norm_e['norm_feat_1_var'] = var

                if self.splitting_rule == "rule_1" :
                    radar_bev_, camera_bev_, pseudo_bev_ = self.splitting(camera_bev_, radar_bev_, pseudo_bev_)

                elif self.splitting_rule == "rule_2":

                    radar_bev_, camera_bev_, pseudo_bev_ = self.splitting_2(camera_bev_, radar_bev_, pseudo_bev_)

                if self.act_mask is not None:
                    feat_bev_ = feat_bev_ * self.act_mask['act_feat_1']
                    camera_bev_ = camera_bev_ * self.act_mask['act_feat_1']
                    radar_bev_ = radar_bev_ * self.act_mask['act_feat_1']
                    pseudo_bev_ = pseudo_bev_ * self.act_mask['act_feat_1']
                else:
                    feat_bev = self.activation(feat_bev_)

        elif self.use_lidar:
            assert(rad_occ_mem0 is not None)
            camera_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)

            pseudo_camera = torch.zeros_like(camera_bev_)
            pseudo_radar = torch.zeros_like(rad_bev_)

            feat_bev_ = torch.cat([camera_bev_, rad_bev_], dim=1)
            camera_bev_ = torch.cat([camera_bev_, pseudo_radar], dim=1)
            radar_bev_ = torch.cat([pseudo_camera, rad_bev_], dim=1)
            pseudo_bev_ = torch.cat([pseudo_camera, pseudo_radar], dim=1)

            feat_bev_ = self.bev_compressor(feat_bev_)
            radar_bev_ = self.bev_compressor(radar_bev_)
            camera_bev_ = self.bev_compressor(camera_bev_)
            pseudo_bev_ = self.bev_compressor(pseudo_bev_)

            radar_bev_ = radar_bev_ - pseudo_bev_
            camera_bev_ = camera_bev_ - pseudo_bev_

            if norm_mask is not None:

                if self.norm_bias_rule == "uniform":

                    feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    radar_bev_ = (radar_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /3) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    camera_bev_ = (camera_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /3) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    pseudo_bev_ = (pseudo_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True) /3) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                elif self.norm_bias_rule == "ratio":
                    cam_mean = torch.mean(camera_bev_, dim=(2,3), keepdim=True)
                    rad_mean = torch.mean(radar_bev_, dim=(2,3), keepdim=True)
                    pseudo_mean = torch.mean(pseudo_bev_, dim=(2,3), keepdim=True)

                    feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    radar_bev_ = (radar_bev_ - rad_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    camera_bev_ = (camera_bev_ - cam_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    pseudo_bev_ = (pseudo_bev_ - pseudo_mean) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

                elif self.norm_bias_rule == "identity" :
                    feat_bev_ = (feat_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    radar_bev_ = (radar_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    camera_bev_ = (camera_bev_) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))
                    pseudo_bev_ = (pseudo_bev_ - torch.mean(feat_bev_, dim=(2,3), keepdim=True)) / torch.sqrt((norm_mask['norm_feat_1_var']+1e-7))

            else:
                mean = torch.mean(feat_bev_, dim=(2,3), keepdim=True)
                var = torch.var(feat_bev_, dim=(2,3), unbiased=False, keepdim=True)
                radar_bev_ = (feat_bev_- mean) / torch.sqrt((var+1e-7))
                norm_e['norm_feat_1_mean'] = mean
                norm_e['norm_feat_1_var'] = var

            if self.splitting_rule == "rule_1" :
                radar_bev_, camera_bev_, pseudo_bev_ = self.splitting(camera_bev_, radar_bev_, pseudo_bev_)

            elif self.splitting_rule == "rule_2":

                radar_bev_, camera_bev_, pseudo_bev_ = self.splitting_2(camera_bev_, radar_bev_, pseudo_bev_)

            if self.conservation_feat_sum :
                diff = feat_bev_ - (camera_bev_ + radar_bev_ + pseudo_bev_)

            if self.act_mask is not None:
                feat_bev_ = feat_bev_ * self.act_mask['act_feat_1']
                camera_bev_ = camera_bev_ * self.act_mask['act_feat_1']
                radar_bev_ = radar_bev_ * self.act_mask['act_feat_1']
                pseudo_bev_ = pseudo_bev_ * self.act_mask['act_feat_1']
            else:
                feat_bev = self.activation(feat_bev_)

        else:
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

            out_dict = self.decoder(feat_bev_, camera_bev_, radar_bev_, pseudo_bev_, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None, self.act_mask, norm_mask)

        act_e = dict()

        if self.act_mask is None:
            act_e['act_feat_1'] = feat_bev / feat_bev_
            act_e['act_feat_2'] = out_dict['act_feat'][0]
            act_e['act_feat_3'] = out_dict['act_feat'][1]
            act_e['act_feat_4'] = out_dict['act_feat'][2]
            act_e['act_feat_5'] = out_dict['act_feat'][3]
            act_e['act_feat_6'] = out_dict['act_feat'][4]
            act_e['act_feat_7'] = out_dict['act_feat'][5]
            act_e['act_feat_8'] = out_dict['act_feat'][6]
            act_e['act_feat_9'] = out_dict['act_feat'][7]
            act_e['act_feat_10'] = out_dict['act_feat'][8]
            act_e['act_feat_11'] = out_dict['act_feat'][9]
            act_e['act_feat_12'] = out_dict['act_feat'][10]
            act_e['act_feat_13'] = out_dict['act_feat'][11]
            act_e['act_feat_14'] = out_dict['act_feat'][12]
            act_e['act_feat_15'] = out_dict['act_feat'][13]

        if norm_mask is None:
            norm_e['norm_feat_2_mean'] = out_dict['norm_feat'][0]
            norm_e['norm_feat_2_var'] = out_dict['norm_feat'][1]
            norm_e['norm_feat_3_mean'] = out_dict['norm_feat'][2]
            norm_e['norm_feat_3_var'] = out_dict['norm_feat'][3]
            norm_e['norm_feat_4_mean'] = out_dict['norm_feat'][4]
            norm_e['norm_feat_4_var'] = out_dict['norm_feat'][5]
            norm_e['norm_feat_5_mean'] = out_dict['norm_feat'][6]
            norm_e['norm_feat_5_var'] = out_dict['norm_feat'][7]

        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation_feat']
        activations = out_dict['up2_activations_cam']
        offset_e = out_dict['instance_offset_cam']

        return act_e, norm_e, seg_e, activations, out_dict

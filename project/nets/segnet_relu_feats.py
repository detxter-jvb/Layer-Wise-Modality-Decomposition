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
    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        )
        # self.key = in_channels

    def forward(self, x, x_skip):
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
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
        self.layer2_1_downsample = self.layer2[0].downsample
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
        self.layer3_1_downsample = self.layer3[0].downsample
        self.layer3_2_conv1 = self.layer3[1].conv1
        self.layer3_2_bn1 = self.layer3[1].bn1
        self.layer3_2_relu = self.layer3[1].relu
        self.layer3_2_conv2 = self.layer3[1].conv2
        self.layer3_2_bn2 = self.layer3[1].bn2
        
        #pdb.set_trace()
        
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
        # self.seg_act = nn.ReLU(inplace=True)
        # self.seg_conv = nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0)
        
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

    def forward(self, x, bev_flip_indices=None, act_mask=None, norm_mask=None):
        b, c, h, w = x.shape

        self.activ_buffer = []
        self.norm_buffer = []
        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)

        if act_mask is not None:
            x = act_mask['act_feat_2'] * x
        else:
            x = self.relu(x)
        self.activ_buffer.append(x > 0)
        
        # (H/4, W/4)
        # x = self.layer1(x)
        identity = x
        out = self.layer1_1_conv1(x)
        out = self.layer1_1_bn1(out)
        if act_mask is not None:
            out = act_mask['act_feat_3'] * out
        else:
            out = self.layer1_1_relu(out)
        self.activ_buffer.append(out > 0)

        out = self.layer1_1_conv2(out)
        out = self.layer1_1_bn2(out)
        out += identity
        if act_mask is not None:
            x = act_mask['act_feat_4'] * out
        else:
            x = self.layer1_1_relu(out)
        self.activ_buffer.append(x > 0)
        
        identity = x
        out = self.layer1_2_conv1(x)
        out = self.layer1_2_bn1(out)

        if act_mask is not None:
            out = act_mask['act_feat_5'] * out
        else:
            out = self.layer1_2_relu(out)
        self.activ_buffer.append(out > 0)

        out = self.layer1_2_conv2(out)
        out = self.layer1_2_bn2(out)
        out += identity

        if act_mask is not None:
            x = act_mask['act_feat_6'] * out
        else:
            x = self.layer1_2_relu(out)
        self.activ_buffer.append(x > 0)

        skip_x['2'] = x
        
        # x = self.layer2(x)
        identity = x
        
        out = self.layer2_1_conv1(x)
        out = self.layer2_1_bn1(out)

        if act_mask is not None:
            out = act_mask['act_feat_7'] * out
        else:
            out = self.layer2_1_relu(out)
        self.activ_buffer.append(out > 0)
        
        out = self.layer2_1_conv2(out)
        out = self.layer2_1_bn2(out)

        identity = self.layer2_1_downsample(x)
        out += identity

        if act_mask is not None:
            x = act_mask['act_feat_8'] * out
        else:
            x = self.layer2_1_relu(out)
        self.activ_buffer.append(x > 0)

        identity = x
        out = self.layer2_2_conv1(x)
        out = self.layer2_2_bn1(out)

        if act_mask is not None:
            out = act_mask['act_feat_9'] * out
        else:
            out = self.layer2_2_relu(out)
        self.activ_buffer.append(out > 0)
        
        out = self.layer2_2_conv2(out)
        out = self.layer2_2_bn2(out)
        out += identity

        if act_mask is not None:
            x = act_mask['act_feat_10'] * out
        else:
            x = self.layer2_2_relu(out)
        self.activ_buffer.append(x > 0)
        
        skip_x['3'] = x
        
        # (H/8, W/8)
        # x = self.layer3(x)
        identity = x
        out = self.layer3_1_conv1(x)
        out = self.layer3_1_bn1(out)

        if act_mask is not None:
            out = act_mask['act_feat_11'] * out
        else:
            out = self.layer3_1_relu(out)

        self.activ_buffer.append(out > 0)
        
        out = self.layer3_1_conv2(out)
        out = self.layer3_1_bn2(out)

        identity = self.layer3_1_downsample(x)
        out += identity
        if act_mask is not None:
            x = act_mask['act_feat_12'] * out
        else:
            x = self.layer3_1_relu(out)
        self.activ_buffer.append(x > 0)
        
        identity = x
        out = self.layer3_2_conv1(x)
        out = self.layer3_2_bn1(out)

        if act_mask is not None:
            out = act_mask['act_feat_13'] * out
        else:
            out = self.layer3_2_relu(out)
        self.activ_buffer.append(out > 0)
        
        out = self.layer3_2_conv2(out)
        out = self.layer3_2_bn2(out)
        out += identity

        if act_mask is not None:
            x = act_mask['act_feat_14'] * out
        else:
            x = self.layer3_2_relu(out)
        self.activ_buffer.append(x > 0)
        
        # First upsample to (H/4, W/4)
        #x, mean_1, var_1 = self.up3_skip(x, skip_x['3'], norm_mask)

        x = self.up3_upsample(x)
        x = self.up3_conv1(x)

        if norm_mask is not None :
            x = (x-norm_mask['norm_feat_2_mean']) / torch.sqrt((norm_mask['norm_feat_2_var']+1e-5))
        else :
            mean = torch.mean(x, dim=(2,3), keepdim=True)
            var = torch.var(x, dim=(2,3), unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var+1e-5)
            self.norm_buffer.append(mean)
            self.norm_buffer.append(var)
        
        x = self.up3_skip(x, skip_x['3'])
        up3_activations = x
        
        # Second upsample to (H/2, W/2)
        #x, mean_2, var_2 = self.up2_skip(x, skip_x['2'], norm_mask)
        x = self.up2_upsample(x)
        x = self.up2_conv1(x)

        if norm_mask is not None :
            x = (x-norm_mask['norm_feat_3_mean']) / torch.sqrt((norm_mask['norm_feat_3_var']+1e-5))
        else :
            mean = torch.mean(x, dim=(2,3), keepdim=True)
            var = torch.var(x, dim=(2,3), unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var+1e-5)
            self.norm_buffer.append(mean)
            self.norm_buffer.append(var)
            
        x = self.up2_skip(x, skip_x['2'])
        up2_activations = x

        
        # Third upsample to (H, W)
        # x, mean_3, var_3 = self.up1_skip(x, skip_x['1'], norm_mask)
        x = self.up1_upsample(x)
        x = self.up1_conv1(x)

        if norm_mask is not None :
            x = (x-norm_mask['norm_feat_4_mean']) / torch.sqrt((norm_mask['norm_feat_4_var']+1e-5))
        else :
            mean = torch.mean(x, dim=(2,3), keepdim=True)
            var = torch.var(x, dim=(2,3), unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var+1e-5)
            self.norm_buffer.append(mean)
            self.norm_buffer.append(var)

        x = self.up1_skip(x, skip_x['1'])
        up1_activations = x
        
        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        
        #segmentation_output = self.seg_conv2(self.seg_act(self.seg_instance_norm(self.seg_conv1(x))))
        x = self.seg_conv1(x)
        
        if norm_mask is not None:
            seg_in = (x - norm_mask['norm_feat_5_mean']) / torch.sqrt(norm_mask['norm_feat_5_var']+1e-5)
        else:
            mean_seg = torch.mean(x, dim=(2,3), keepdim=True)
            var_seg = torch.var(x, dim=(2,3), unbiased=False, keepdim=True)
            self.norm_buffer.append(mean_seg)
            self.norm_buffer.append(var_seg)
            seg_in = (x - mean_seg) / torch.sqrt(var_seg+1e-5)
        
        if act_mask is not None:
            seg_out = act_mask['act_feat_15'] * seg_in
        else:
            seg_out = self.seg_act(seg_in)
        self.activ_buffer.append(seg_out > 0)
        segmentation_output = self.seg_conv2(seg_out)
        #segmentation_output = self.segmentation_head(x)
        
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'act_feat': self.activ_buffer,
            'norm_feat' : self.norm_buffer,
            'up3_activations' : up3_activations, 
            'up2_activations' : up2_activations,
            'up1_activations' : up1_activations,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
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

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

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

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x

class Segnet_relu_feats(nn.Module):
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
        super(Segnet_relu_feats, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

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
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        if self.use_radar and self.use_lidar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y +Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                )
        elif self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    #nn.InstanceNorm2d(latent_dim)
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
                # use simple sum
                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # set_bn_momentum(self, 0.1)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None, act_mask=None, norm_mask=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        self.act_mask = act_mask
        self.norm_mask = norm_mask
        
        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
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

        # unproject image feature to 3d grid
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
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        norm_e = dict()
        
        if self.use_radar and self.use_lidar:
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            lid_bev_ = rad_occ_mem0[0].permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            rad_bev_ = rad_occ_mem0[1].permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_, lid_bev_], dim=1)
            feat_bev_ = self.bev_compressor(feat_bev_)

            if self.norm_mask is not None:
                feat_bev_ = (feat_bev_ - self.norm_mask['norm_feat_1_mean']) / torch.sqrt((self.norm_mask['norm_feat_1_var']+1e-5))
            else:
                mean = torch.mean(feat_bev_, dim=(2,3), keepdim=True)
                var = torch.var(feat_bev_, dim=(2,3), unbiased=False, keepdim=True)
                feat_bev_ = (feat_bev_- mean) / torch.sqrt((var+1e-5))
                norm_e['norm_feat_1_mean'] = mean
                norm_e['norm_feat_1_var'] = var
                # implement hands-on instance normalization
            if self.act_mask is not None:
                feat_bev = feat_bev_ * self.act_mask['act_feat_1']
            else:
                feat_bev = self.activation(feat_bev_)
        elif self.use_radar:
            assert(rad_occ_mem0 is not None)
            if not self.use_metaradar:
                print("should not be inside here")
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev_ = self.bev_compressor(feat_bev_)                    
                if self.act_mask is not None:
                    feat_bev = feat_bev_ * self.act_mask['act_feat_1']
                else:
                    feat_bev = self.activation(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
                #feat_bev_ = torch.zeros_like(feat_bev_)
                #rad_bev_ = torch.zeros_like(rad_bev_)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev_ = self.bev_compressor(feat_bev_)

                if self.norm_mask is not None:
                    feat_bev_ = (feat_bev_ - self.norm_mask['norm_feat_1_mean']) / torch.sqrt((self.norm_mask['norm_feat_1_var']+1e-5))
                else:
                    mean = torch.mean(feat_bev_, dim=(2,3), keepdim=True)
                    var = torch.var(feat_bev_, dim=(2,3), unbiased=False, keepdim=True)
                    feat_bev_ = (feat_bev_- mean) / torch.sqrt((var+1e-5))
                    norm_e['norm_feat_1_mean'] = mean
                    norm_e['norm_feat_1_var'] = var
                    # implement hands-on instance normalization
                    
                if self.act_mask is not None:
                    feat_bev = feat_bev_ * self.act_mask['act_feat_1']
                else:
                    feat_bev = self.activation(feat_bev_)
                
        elif self.use_lidar:
            assert(rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else: # rgb only
            #print("check happening?")
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None, self.act_mask, self.norm_mask)

        act_e = dict()
        #print("check out_dict length:", len(out_dict['act_feat']))
        epsilon = 1e-6
        if self.act_mask is None: 
            act_e['act_feat_1'] = feat_bev / (feat_bev_ + epsilon)
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

        if self.norm_mask is None:
            norm_e['norm_feat_2_mean'] = out_dict['norm_feat'][0]
            norm_e['norm_feat_2_var'] = out_dict['norm_feat'][1]
            norm_e['norm_feat_3_mean'] = out_dict['norm_feat'][2]
            norm_e['norm_feat_3_var'] = out_dict['norm_feat'][3]
            norm_e['norm_feat_4_mean'] = out_dict['norm_feat'][4]
            norm_e['norm_feat_4_var'] = out_dict['norm_feat'][5]
            norm_e['norm_feat_5_mean'] = out_dict['norm_feat'][6]
            norm_e['norm_feat_5_var'] = out_dict['norm_feat'][7]

        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        activations = out_dict['up2_activations']
        offset_e = out_dict['instance_offset']

        return act_e, norm_e, seg_e, activations, offset_e

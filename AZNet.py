import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from ResNet import Backbone_ResNet50_in3
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import math
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CSAFM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CSAFM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels,in_channels,kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,in_channels,kernel_size=1),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.BatchNorm2d(in_channels),
        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channels*3)
        self.r1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,dilation=1,stride=1)
        self.r3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=3,dilation=3,stride=1)
        self.r5 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=5,dilation=5,stride=1)
        self.bcr = nn.Sequential(
            nn.Conv2d(in_channels*3,in_channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
    def forward(self,x_cur,x_lat,pre=1):
        if pre==1:
            x_lat = self.conv2(x_lat)

        elif pre==2:
            x_lat = self.conv1(x_lat)

        x_mul = x_cur * x_lat
        x_mul = torch.mul(self.sa(x_mul), x_cur)
        x_cur = self.r1(x_cur)
        x_mul = self.r3(x_mul)
        x_lat = self.r5(x_lat)
        x_all = torch.cat((x_cur, x_mul, x_lat),dim=1)
        x_all_sum = torch.mul(self.ca(x_all), x_all)
        x_all_sum = self.bcr(x_all_sum)
        return x_all_sum

class self_attention(nn.Module):
    def __init__(self,in_channels):
        super(self_attention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(input).view(batch_size, -1, height * width)
        v = self.value(input).view(batch_size, -1, height * width)
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)

        return self.gamma * out + input
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # need modify by the batchsize/GPU_number
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ChannelAttention_diag(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=2):
        super(ChannelAttention_diag, self).__init__()
        self.inter_channels = in_channels // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_channels, self.inter_channels, bias=False),
                           nn.ReLU(inplace=True))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_channels, self.inter_channels, bias=False),
                           nn.ReLU(inplace=True))
    def forward(self, ftr):
        device = torch.device("cpu")
        B, C, H, W = ftr.size()
        M = self.inter_channels
        ftr_avg = self.avg_fc(self.avg_pool(ftr).squeeze())
        ftr_max = self.max_fc(self.max_pool(ftr).squeeze())
        cw = torch.sigmoid(ftr_avg + ftr_max).unsqueeze(-1)
        b = torch.unsqueeze(torch.eye(M, device=device), 0).expand(B, M, M)
        return torch.mul(b, cw)

class GR(nn.Module):
    def __init__(self, in_channels, node_n, squeeze_ratio=2):
        super(GR, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.conv_k = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                      nn.ReLU(inplace=True))
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.ca_diag = ChannelAttention_diag(in_channels)
        self.GCN = GraphConvolution(in_channels, in_channels, bias=False)
        self.delta = nn.Parameter(torch.Tensor([0]))

    def forward(self, ftr):
        device = torch.device("cpu")
        B, C, H, W = ftr.size()
        HW = H * W
        M = C // 8
        b = torch.unsqueeze(torch.eye(HW, device=device), 0).expand(B, HW, HW)
        One = torch.ones(HW, 1, dtype=torch.float32, device=device).expand(B, HW, 1)
        diag = self.ca_diag(ftr)

        ftr_k = self.conv_k(ftr).view(B, -1, HW)
        ftr_q = ftr_k.permute(0, 2, 1)

        D = torch.bmm(ftr_q, diag)
        D = torch.sigmoid(torch.bmm(D, ftr_k))
        D = torch.bmm(D, One)
        D = D ** (-1 / 2)
        D = torch.mul(b, D)

        P = torch.bmm(D, ftr_q)
        Pt = P.permute(0, 2, 1)

        X = ftr.view(B, -1, HW).permute(0, 2, 1)
        LX = torch.bmm(Pt, X)
        LX = torch.bmm(diag, LX)
        LX = torch.bmm(P, LX)
        LX = X - LX

        Y = (X + self.GCN(LX)).permute(0, 2, 1)

        Y = Y.view(B, C, H, W)
        return Y

class Graph_aspp(nn.Module):
    def __init__(self,in_channels):
        super(Graph_aspp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels//4,in_channels,kernel_size=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d((in_channels//4)*3,in_channels,kernel_size=1),
        )
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.down = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1)
        self.rate0 = nn.Conv2d(in_channels,in_channels//4,kernel_size=3,stride=1,padding=1,dilation=1)
        self.rate1 = nn.Conv2d(in_channels,in_channels//4,kernel_size=3,stride=1,padding=2,dilation=2)
        self.rate2 = nn.Conv2d(in_channels,in_channels//4,kernel_size=3,stride=1,padding=4,dilation=4)

        self.graph1 = GR(in_channels//4,16*16)
        self.graph2 = GR(in_channels//4,8*8)

        self.att = self_attention(in_channels)
    def forward(self,x_cur,x_lat,pre=1):
        if pre == 1:
            x_sum = x_cur * self.up(x_lat) + x_cur

            feature1 = self.rate0(x_sum) + self.conv(x_sum)
            x_1 = self.graph1(feature1)
            feature1 = self.conv1(feature1)

            feature = feature1 + x_sum
            feature2 = self.rate1(feature) + self.conv(feature)
            x_2 = self.graph1(feature2)
            feature2 = self.conv1(feature2)

            feature = feature2 + x_sum
            feature3 = self.rate2(feature) + self.conv(feature)
            x_3 = self.graph1(feature3)
            cat = torch.cat((x_1,x_2,x_3),dim=1)
            cat = self.conv3(cat)
        else:
            x_sum = x_lat * self.down(x_cur) + x_lat

            feature1 = self.rate0(x_sum) + self.conv(x_sum)
            x_1 = self.graph2(feature1)
            feature1 = self.conv1(feature1)

            feature = feature1 + x_sum
            feature2 = self.rate1(feature) + self.conv(feature)
            x_2 = self.graph2(feature2)
            feature2 = self.conv1(feature2)

            feature = feature2 + x_sum
            feature3 = self.rate2(feature) + self.conv(feature)
            x_3 = self.graph2(feature3)
            feature3 = self.conv1(feature3)

            feature = feature3 + x_sum
            feature4 = self.rate2(feature) + self.conv(feature)
            x_4 = self.graph2(feature4)
            cat = torch.cat((x_1,x_2,x_3,x_4),dim=1)

        return cat

class Sp_F(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sp_F, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                      padding=1, dilation=1, bias=False),
        )
        self.conv1 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=1)

    def forward(self, x_cur, x_lat):
        x_mul = self.conv(x_cur) * self.conv1(x_lat)
        x_sum = x_mul + self.conv(x_cur)
        return x_sum


class Se_F(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Se_F, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_1D = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.conv_2D_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_2D_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x_cur, x_lat):
        x_cur = self.up(x_cur)
        x_cur = self.conv_1D(x_cur)
        x_mul_1 = self.conv_2D_1(x_cur) * self.conv_2D_2(x_lat)
        x_mul_2 = torch.mul(x_mul_1, x_lat)
        x_mul_3 = x_mul_2 + x_lat
        return x_mul_3


class fuse_module_1(nn.Module):
    def __init__(self, in_channels):
        super(fuse_module_1, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channels)
        self.bcr = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
    def forward(self, x1, x2, x3):
        x_sum = torch.cat((x1,x3),dim=1)
        x_sum = self.bcr(x_sum)
        x_sum_ca = self.ca(x_sum)
        x_sum_sa = self.sa(x_sum)

        x_mul1 = x2 * x_sum_ca
        x_mul2 = x2 * x_sum_sa
        x_mul_all = x_mul1 + x_mul2 + x2
        return x_mul_all


class HFAF(nn.Module):
    def __init__(self):
        super(HFAF, self).__init__()
        self.sp_f_1 = Sp_F(64, 128)
        self.sp_f_2 = Sp_F(128, 256)
        self.se_f_1 = Se_F(512, 256)
        self.se_f_2 = Se_F(512, 512)
        self.fuse = fuse_module_1(256)

    def forward(self, x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb):
        p_f1 = self.sp_f_1(x1_rgb, x2_rgb)
        p_f2 = self.sp_f_2(x2_rgb, x3_rgb)
        p_f3 = self.sp_f_2(p_f1,p_f2)

        e_f1 = self.se_f_2(x5_rgb, x4_rgb)
        e_f2 = self.se_f_1(x4_rgb, x3_rgb)
        e_f3 = self.se_f_1(e_f1,e_f2)

        fuse = self.fuse(p_f3, x3_rgb, e_f3)
        return fuse

class fuse_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fuse_module, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.CBR(x)
        x = self.upsample(x)
        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.fuse5 = fuse_module(512, 512)
        self.fuse4 = fuse_module(512, 256)
        self.fuse3 = fuse_module(256, 128)
        self.fuse2 = fuse_module(128, 64)
        self.fuse1 = fuse_module(64, 32)


        self.S5 = nn.Conv2d(512, 2, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(256, 2, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(128, 2, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(32, 2, 3, stride=1, padding=1)

    def forward(self, x1_Accom, x2_Accom, x3_att1, x4_sma, x5_sma, x5_rgb, x4_rgb, x3_rgb, x2_rgb, x1_rgb):
        z5 = torch.mul(x5_rgb, x5_sma) + x5_rgb
        x5 = self.fuse5(z5)
        t5 = self.S5(x5)

        z4 = torch.mul(x4_sma, x5) + x5 + x4_rgb
        x4 = self.fuse4(z4)
        t4 = self.S4(x4)

        z3 = x4 + x3_rgb + x3_att1
        x3 = self.fuse3(z3)
        t3 = self.S3(x3)

        z2 = x3 + x2_Accom + x2_rgb
        x2 = self.fuse2(z2)
        t2 = self.S2(x2)

        z1 = x2 + x1_Accom + x1_rgb
        x1 = self.fuse1(z1)
        t1 = self.S1(x1)

        return t1

class MyConv_resnet(nn.Module):
    def __init__(self):
        super(MyConv_resnet, self).__init__()
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_ResNet50_in3()

        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(256, 128, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(512, 256, 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(1024, 512, 3, stride=1, padding=1)
        self.lateral_conv4 = BasicConv2d(2048, 512, 3, stride=1, padding=1)

        self.Accom1 = CSAFM(64,128)
        self.Accom2 = CSAFM(128,64)

        self.att1 = Graph_aspp(512)
        self.att2 = Graph_aspp(512)

        self.AAPP1 = HFAF()

        self.decoder_rgb = decoder()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_rgb):
        x0 = self.encoder1(x_rgb)
        y0 = x0
        x1 = self.encoder2(x0)
        y1 = x1
        x2 = self.encoder4(x1)
        y2 = x2
        x3 = self.encoder8(x2)
        y3 = x3
        x4 = self.encoder16(x3)
        y4 = x4

        x1_rgb = self.lateral_conv0(x0)
        x2_rgb = self.lateral_conv1(x1)
        x3_rgb = self.lateral_conv2(x2)
        x4_rgb = self.lateral_conv3(x3)
        x5_rgb = self.lateral_conv4(x4)

        x1_Accom = self.Accom1(x1_rgb,x2_rgb,pre=1)
        x2_Accom = self.Accom2(x2_rgb,x1_rgb,pre=2)

        x3_att1 = self.AAPP1(x1_rgb,x2_rgb,x3_rgb,x4_rgb,x5_rgb)

        x4_sma = self.att1(x4_rgb,x5_rgb,pre=1)
        x5_sma = self.att2(x4_rgb,x5_rgb,pre=2)
        s1 = self.decoder_rgb(x1_Accom,x2_Accom,x3_att1,x4_sma,x5_sma,x5_rgb,x4_rgb,x3_rgb,x2_rgb,x1_rgb)

        return s1

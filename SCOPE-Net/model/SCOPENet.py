import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from mmengine.model import constant_init
from einops import rearrange
import typing as t


from typing import List, Callable
from torch import Tensor

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,activation='relu'):
        super(BasicConv2d, self).__init__()
        self.activation=activation
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

# SAM
class SAM(nn.Module):
    def __init__(self, in_ch, out_ch,spatial_kernel_sizes=[3, 5, 7, 9],channel_kernel_sizes=[7, 11, 21],heads = [1,8]):
        super(SAM,self).__init__()
        self.lrssa = LRSSA(in_ch,spatial_kernel_sizes)
        self.mscsa = MSCSA(in_ch,channel_kernel_sizes,heads)
        self.mscb = MSCB(in_ch, out_ch)
      
    def forward(self, x):
        x = self.lrssa(x)
        x = self.mscsa(x)
        x = self.mscb(x)
        return x
    
class LRSSA(nn.Module):
    def __init__(self, in_ch,spatial_kernel_sizes):
        super(LRSSA,self).__init__()
        self.in_ch=in_ch 
        self.dynamic_convs = nn.ModuleList([
                DynamicConv1d(in_ch // 4, in_ch // 4, spatial_kernel_sizes)
                for _ in range(4)  
        ])
        self.hnorm = nn.GroupNorm(4, in_ch)
        self.wnorm = nn.GroupNorm(4, in_ch)
        self.sig = nn.Sigmoid()

      
    def forward(self, x):
        b, c, h, w = x.size()
        hx = x.mean(dim=3)  # [B,C,H]
        wx = x.mean(dim=2)  # [B,C,W]
        
        hx_parts = torch.split(hx, self.in_ch//4, dim=1)
        wx_parts = torch.split(wx, self.in_ch//4, dim=1)
        
        hx_out = []
        wx_out = []

        for i in range(4):
            hx_out.append(self.dynamic_convs[i](hx_parts[i]))  
            wx_out.append(self.dynamic_convs[i](wx_parts[i]))
        hx_attn = self.sig(self.hnorm(torch.cat(hx_out, dim=1)))
        wx_attn = self.sig(self.wnorm(torch.cat(wx_out, dim=1)))
        hx_attn = hx_attn.view(b, c, h, 1)
        wx_attn = wx_attn.view(b, c, 1, w)
        return x * hx_attn * wx_attn
    
class DynamicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.experts = len(kernel_sizes)
        self.conv_kernels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, ks, 
                      padding=ks//2, groups=in_channels)
            for ks in kernel_sizes
        ])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, self.experts, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # x: [B, C, L]
        attn_weights = self.attn(x)  # [B, experts]
        outputs = 0.0
        for i in range(self.experts):
            w = attn_weights[:, i].view(-1, 1, 1)  # [B,1,1]
            outputs += w * self.conv_kernels[i](x)
        return outputs

class MSCSA(nn.Module):
    def __init__(self, in_ch,channel_kernel_sizes=[7, 11, 21],heads = [1,8]):
        super(MSCSA,self).__init__()
        self.lrb = SHSA(in_ch, channel_kernel_sizes, heads[0], in_ch // heads[0])
        self.srb = MHSA(in_ch, channel_kernel_sizes, heads[1], in_ch // heads[1])
        self.weights = nn.Parameter(torch.ones(4))  
        self.sig = nn.Sigmoid()
   
    def forward(self, x):
        lrb_h, lrb_w= self.lrb(x)
        srb_h, srb_w= self.srb(x)
        attn = (self.weights[0] * lrb_h + self.weights[1] * lrb_w + self.weights[2] * srb_h + self.weights[3] * srb_w)
        attn = self.sig(attn)
        return attn * x

class SHSA(nn.Module):
    def __init__(self, in_ch, channel_kernel_sizes,heads, head_dim):
        super(SHSA,self).__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scaler=self.head_dim ** -0.5 
        self.q_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[0],1), padding=(3, 0),bias=False, groups=in_ch)
        self.k_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[1],1), padding=(5, 0),bias=False, groups=in_ch)
        self.v_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[2],1), padding=(10, 0),bias=False, groups=in_ch)
        self.q_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[0]), padding=(0, 3),bias=False, groups=in_ch)
        self.k_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[1]), padding=(0, 5),bias=False, groups=in_ch)
        self.v_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[2]), padding=(0, 10),bias=False, groups=in_ch)
      
    def forward(self, x):
        _, _, h, w = x.size()
        q_h = self.q_h(x)
        k_h = self.k_h(x)
        v_h = self.v_h(x)
        q_w = self.q_w(x)
        k_w = self.k_w(x)
        v_w = self.v_w(x)

        q_h = rearrange(q_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_h = rearrange(k_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_h = rearrange(v_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)

        attn_h = q_h @ k_h.transpose(-2, -1)
        attn_h = attn_h * self.scaler
        attn_h = attn_h.softmax(dim=-1)
        attn_h = attn_h @ v_h
        attn_h = rearrange(attn_h, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_h = attn_h.mean((2, 3), keepdim=True)

        q_w = rearrange(q_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_w = rearrange(k_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_w = rearrange(v_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        
        attn_w = q_w @ k_w.transpose(-2, -1)
        attn_w = attn_w * self.scaler 
        attn_w = attn_w.softmax(dim=-1)
    
        attn_w = attn_w @ v_w
        attn_w = rearrange(attn_w, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_w = attn_w.mean((2, 3), keepdim=True)

        return attn_h,attn_w
    
class MHSA(nn.Module):
    def __init__(self, in_ch, channel_kernel_sizes, heads, head_dim):
        super(MHSA,self).__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scaler=self.head_dim ** -0.5 
        self.q_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[0],1), padding=(3, 0),bias=False, groups=in_ch)
        self.k_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[1],1), padding=(5, 0),bias=False, groups=in_ch)
        self.v_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[2],1), padding=(10, 0),bias=False, groups=in_ch)
        self.q_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[0]), padding=(0, 3),bias=False, groups=in_ch)
        self.k_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[1]), padding=(0, 5),bias=False, groups=in_ch)
        self.v_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[2]), padding=(0, 10),bias=False, groups=in_ch)
    
    def forward(self, x):
        _, _, h, w = x.size()

        q_h = self.q_h(x)
        k_h = self.k_h(x)
        v_h = self.v_h(x)
        q_w = self.q_w(x)
        k_w = self.k_w(x)
        v_w = self.v_w(x)

        q_h = rearrange(q_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_h = rearrange(k_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_h = rearrange(v_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)

        attn_h = q_h @ k_h.transpose(-2, -1)
        attn_h = attn_h * self.scaler
        attn_h = attn_h.softmax(dim=-1)
        attn_h = attn_h @ v_h
        attn_h = rearrange(attn_h, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_h = attn_h.mean((2, 3), keepdim=True)

        q_w = rearrange(q_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_w = rearrange(k_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_w = rearrange(v_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        
        attn_w = q_w @ k_w.transpose(-2, -1)
        attn_w = attn_w * self.scaler 
        attn_w = attn_w.softmax(dim=-1)
    
        attn_w = attn_w @ v_w
        attn_w = rearrange(attn_w, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_w = attn_w.mean((2, 3), keepdim=True)

        return attn_h,attn_w

class MSCB(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_sizes=[3,5,7]):
        super(MSCB, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_ch, out_planes=in_ch*6, kernel_size=1)
        self.msdconv = msdconv(in_ch*6,kernel_sizes)
        self.pwconv = BasicConv2d(in_ch*6, out_ch, kernel_size=1,activation=None)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        c1 = self.conv1(x)
        msdc1 = self.msdconv(c1)
        out = self.pwconv(msdc1)
        return x + out
    
class msdconv(nn.Module):
    def __init__(self, in_ch, kernel_sizes):
        super(msdconv, self).__init__()
        self.dconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size,1, kernel_size // 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x1,x2,x3=self.dconvs[0](x),self.dconvs[1](x),self.dconvs[2](x)
        return x1+x2+x3

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners) 

#DCERM
class DCERM(nn.Module):
    def __init__(self, x1_ch, x2_ch, x3_ch, x4_ch, out_ch):
        super(DCERM, self).__init__()
        self.threeD_Conv = ThreeD_Conv(x2_ch, x3_ch, x4_ch, x1_ch)
        self.asee = ASEE(in_dim=64,hidden_dim=64)
        self.desc=DESC(in_ch = x1_ch)
        self.conv = BasicConv2d(x1_ch, out_ch, kernel_size=1,activation='silu')

    def forward(self, x1, x2, x3, x4):
        semantic_feature=self.threeD_Conv(x2, x3, x4)
        semantic_feature=F.interpolate(semantic_feature, scale_factor=2, mode='bilinear', align_corners=False)
        edge_feature=self.asee(x1)
        edge_feature, semantic_feature= self.desc(edge_feature,semantic_feature)
        out = self.conv(torch.cat([edge_feature, semantic_feature], dim=1))
        return out
    
class ThreeD_Conv(nn.Module):
    def __init__(self, x2_ch, x3_ch, x4_ch, out_ch):
        super(ThreeD_Conv, self).__init__()
        self.conv1 = BasicConv2d(x2_ch, out_ch, kernel_size=1,activation='silu')
        self.conv2 = BasicConv2d(x3_ch, out_ch, kernel_size=1,activation='silu')
        self.conv3 = BasicConv2d(x4_ch, out_ch, kernel_size=1,activation='silu')
        self.conv3d = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3),padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.avgpool_3d = nn.AvgPool3d(kernel_size=(3, 1, 1))

    def forward(self, x2, x3, x4):
        x2 = self.conv1(x2)
        x3 = self.conv2(x3)
        x3 = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x4 = self.conv3(x4)
        x4 = F.interpolate(x4, x2.size()[2:], mode='nearest')
        x2_3d = torch.unsqueeze(x2, -3)
        x3_3d = torch.unsqueeze(x3, -3)
        x4_3d = torch.unsqueeze(x4, -3)
        x_fuse = torch.cat([x2_3d, x3_3d, x4_3d], dim=2)
        x_fuse_3d = self.conv3d(x_fuse)
        x_fuse_bn = self.bn(x_fuse_3d)
        x_act = self.leakyrelu(x_fuse_bn)
        x = self.avgpool_3d(x_act)
        x = torch.squeeze(x, 2)
        return x
    
class ASEE(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super(ASEE, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(in_dim,hidden_dim, 3, padding=1, bias=False),
            norm(hidden_dim),
            act()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.dpp = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                DeformConv2d(inc = hidden_dim, outc = hidden_dim, kernel_size = 3),
                norm(hidden_dim),
                nn.SiLU(inplace=True)
                ))
            self.dpp.append(dpp(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * (width-1), hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        mid = self.in_conv(x)
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)
            if i == 0:
                out = self.dpp[i](mid)
            else:
                out = torch.cat([out, self.dpp[i](mid)], dim=1)
        out = self.out_conv(out)
        out= x + out
        return out
    
class dpp(nn.Module):
    def __init__(self, in_dim, norm, act):
        super(dpp, self).__init__()
        self.avgpool = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        edge = x_avg - x_max
        edge = self.out_conv(edge)
        return x+edge

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
   
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias) 

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0) 
        self.p_conv.register_backward_hook(self._set_lr)
        
        self.modulation = modulation
        if modulation: 
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr) 

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation: 
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
 
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()         
        q_rb = q_lt + 1                 

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)     
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)     

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))    
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))   
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))  
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))  

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)      
        x_q_rb = self._get_x_q(x, q_rb, N)     
        x_q_lb = self._get_x_q(x, q_lb, N)     
        x_q_rt = self._get_x_q(x, q_rt, N)     

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt


        if self.modulation: # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1) # (b,h,w,N)
            m = m.unsqueeze(dim=1) # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1) # (b,c,h,w,N)
            x_offset *= m 

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)   
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DESC(nn.Module):
    def __init__(self, in_ch):
        super(DESC,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, in_ch//2, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

        self.conv3 = BasicConv2d(in_ch // 2, in_ch // 2, 1)
        self.conv4 = BasicConv2d(in_ch // 2, in_ch // 2, 1)

        self.w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.silu = nn.SiLU(inplace=True)
        self.conv5 = nn.Conv2d(in_ch//2 , in_ch // 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_ch//2 , in_ch // 2, kernel_size=1, stride=1, padding=0)
      
    def forward(self, x1, semantic_feature):
        edge_feature = self.conv1(x1)
        semantic_feature = self.conv2(semantic_feature)
        
        edge_feature_sig = self.Sigmoid(edge_feature)          #32x88x88 
        semantic_feature_sig = self.Sigmoid(semantic_feature)

        edge_feature = self.conv3(edge_feature)
        semantic_feature = self.conv4(semantic_feature)

        w1 = self.w1
        w2 = self.w2
        
        weight1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)        
        edge_feature_1 = self.silu(self.conv5(weight1[0]*edge_feature + weight1[1]*(edge_feature * edge_feature_sig) + weight1[2]*((1 - edge_feature_sig) * semantic_feature_sig * semantic_feature)))
        semantic_feature_1 = self.silu(self.conv6(weight2[0]*semantic_feature + weight2[1]*(semantic_feature * semantic_feature_sig) + weight2[2]*((1 - semantic_feature_sig) * edge_feature_sig * edge_feature)))

        return edge_feature_1, semantic_feature_1

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel//reduction, in_channel, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg = self.conv(self.avg_pool(x))
        max = self.conv(self.max_pool(x))
        att = self.sigmoid(avg + max)
        return x * att
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return self.sigmoid(attn) * x 

class PDecoder(nn.Module):
    def __init__(self, channel):
        super(PDecoder, self).__init__()
        self.relu = nn.ReLU(True)                       
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.ca_x2 = ChannelAttention(channel)
        self.ca_x3 = ChannelAttention(channel)
        self.ca_x4 = ChannelAttention(channel)
        self.ca_x5 = ChannelAttention(channel)
        self.ca_x6 = ChannelAttention(2*channel)
        self.ca_x7 = ChannelAttention(3*channel)
        self.ca_x8 = ChannelAttention(4*channel)
        self.ca_x9 = ChannelAttention(5*channel)
        
        self.sa_x2 = SpatialAttention()
        self.sa_x3 = SpatialAttention()
        self.sa_x4 = SpatialAttention()
        self.sa_x5 = SpatialAttention()
        self.sa_x6 = SpatialAttention()
        self.sa_x7 = SpatialAttention()
        self.sa_x8 = SpatialAttention()
        self.sa_x9 = SpatialAttention()

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample10 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample11 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample12 = BasicConv2d(2*channel, 2*channel, 3, padding=1, activation='silu')
        self.conv_upsample13 = BasicConv2d(3*channel, 3*channel, 3, padding=1, activation='silu')
        self.conv_upsample14 = BasicConv2d(4*channel, 4*channel, 3, padding=1, activation='silu')
        self.conv_concat1 = BasicConv2d(2*channel, 2*channel, 3, padding=1, activation='silu')
        self.conv_concat2 = BasicConv2d(3*channel, 3*channel, 3, padding=1, activation='silu')
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1, activation='silu')
        self.conv_concat4 = BasicConv2d(5*channel, 5*channel, 3, padding=1, activation='silu')
        self.conv4 = BasicConv2d(5*channel, 5*channel, 3, padding=1, activation='silu')
        self.conv5 = nn.Conv2d(5*channel, 1, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x1_1 = x1 
        
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x2_1 = self.ca_x2(x2_1)  
        x2_1 = self.sa_x2(x2_1)  

        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * \
               self.conv_upsample3(self.upsample(x2)) * x3
        x3_1 = self.ca_x3(x3_1)  
        x3_1 = self.sa_x3(x3_1) 

        x4_1 = self.conv_upsample4(self.upsample(self.upsample(self.upsample(x1)))) * \
               self.conv_upsample5(self.upsample(self.upsample(x2))) * \
               self.conv_upsample6(self.upsample(x3)) * x4
        x4_1 = self.ca_x4(x4_1) 
        x4_1 = self.sa_x4(x4_1) 

        x5_1 = self.conv_upsample7(self.upsample(self.upsample(self.upsample(x1)))) * \
               self.conv_upsample8(self.upsample(self.upsample(x2))) * \
               self.conv_upsample9(self.upsample(x3)) * \
               self.conv_upsample10(x4) * x5
        x5_1 = self.ca_x5(x5_1)  
        x5_1 = self.sa_x5(x5_1)          

        x2_2 = torch.cat((x2_1, self.conv_upsample11(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat1(x2_2)
        x2_2 = self.ca_x6(x2_2)  
        x2_2 = self.sa_x6(x2_2)  

        x3_2 = torch.cat((x3_1, self.conv_upsample12(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat2(x3_2)
        x3_2 = self.ca_x7(x3_2) 
        x3_2 = self.sa_x7(x3_2)  

        x4_2 = torch.cat((x4_1, self.conv_upsample13(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat3(x4_2)
        x4_2 = self.ca_x8(x4_2) 
        x4_2 = self.sa_x8(x4_2) 

        x5_2 = torch.cat((x5_1, self.conv_upsample14(x4_2)), 1)
        x5_2 = self.conv_concat4(x5_2)
        x5_2 = self.ca_x9(x5_2) 
        x5_2 = self.sa_x9(x5_2) 

        x = self.conv4(x5_2)
        x = self.conv5(x)
        return x

class Upmodel(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(Upmodel, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,     
            stride=1,      
            padding=1,
            bias=bias      
        )
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        out = self.upsample(x)
        return out


class SCOPENet(nn.Module):
    def __init__(self, channel=32):
        super(SCOPENet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './model/pvt_v2_b2.pth'       
        save_model = torch.load(path) 
        model_dict = self.backbone.state_dict()         
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}           
        model_dict.update(state_dict)                   
        self.backbone.load_state_dict(model_dict)       

        self.ChannelReduction_1 = BasicConv2d(64, channel, 3, 1, 1, activation=None)  # 64x88x88->32x88x88           
        self.ChannelReduction_2 = BasicConv2d(128, channel, 3, 1, 1, activation=None) # 128x44x44->32x44x44
        self.ChannelReduction_3 = BasicConv2d(320, channel, 3, 1, 1, activation=None) # 320x22x22->32x22x22
        self.ChannelReduction_4 = BasicConv2d(512, channel, 3, 1, 1, activation=None) # 512x11x11->32x11x11

        self.sam1=SAM(in_ch=32, out_ch=32)
        self.sam2=SAM(in_ch=32, out_ch=32)
        self.sam3=SAM(in_ch=32, out_ch=32)   
        self.sam4=SAM(in_ch=32, out_ch=32)

        self.dcerm = DCERM(x1_ch=64, x2_ch =128, x3_ch=320, x4_ch=512, out_ch=32)

        self.PDecoder = PDecoder(channel)   
        self.upmodel = Upmodel(in_ch=1, out_ch=1)
        self.sigmoid = nn.Sigmoid()        


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x5=self.dcerm(x1,x2,x3,x4)            #32x88x88

        x1_cr = self.ChannelReduction_1(x1) # 32x88x88
        x2_cr = self.ChannelReduction_2(x2) # 32x44x44
        x3_cr = self.ChannelReduction_3(x3) # 32x22x22
        x4_cr = self.ChannelReduction_4(x4) # 32x11x11

        x1=self.sam1(x1_cr)
        x2=self.sam2(x2_cr)     
        x3=self.sam3(x3_cr)
        x4=self.sam4(x4_cr)

        prediction = self.upmodel(self.PDecoder(x4, x3, x2, x1, x5))
         
        return prediction, self.sigmoid(prediction)
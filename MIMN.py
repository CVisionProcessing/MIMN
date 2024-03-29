from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
from models.Fusion import InvSKAttention
from models.memory import USEMemory, updateMemory, chooseMemory
from fightingcv_attention.conv.Involution import Involution

import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, group=1):
        super(Encoder, self).__init__()
        
        resnet_im = models.resnet101(pretrained=True)
        self.conv1_1 = resnet_im.conv1
        self.bn1_1 = resnet_im.bn1
        self.relu_1 = resnet_im.relu
        self.maxpool_1 = resnet_im.maxpool

        self.res2_1 = resnet_im.layer1
        self.res3_1 = resnet_im.layer2
        self.res4_1 = resnet_im.layer3
        self.res5_1 = resnet_im.layer4

        resnet_fl = models.resnet101(pretrained=True)
        self.conv1_2 = resnet_fl.conv1
        self.bn1_2 = resnet_fl.bn1
        self.relu_2 = resnet_fl.relu
        self.maxpool_2 = resnet_fl.maxpool

        self.res2_2 = resnet_fl.layer1
        self.res3_2 = resnet_fl.layer2
        self.res4_2 = resnet_fl.layer3
        self.res5_2 = resnet_fl.layer4
        
        self.sk3 = InvSKAttention(channel=512,reduction=8)
        self.sk4 = InvSKAttention(channel=1024,reduction=16)
        self.sk5 = InvSKAttention(channel=2048,reduction=32)
        
    def forward_res2(self, f1, f2):
        x1 = self.conv1_1(f1)
        x1 = self.bn1_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.maxpool_1(x1)
        r2_1 = self.res2_1(x1)

        x2 = self.conv1_2(f2)
        x2 = self.bn1_2(x2)
        x2 = self.relu_2(x2)
        x2 = self.maxpool_2(x2)
        r2_2 = self.res2_2(x2)

        return r2_1, r2_2
    
    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        r2_1, r2_2 = self.forward_res2(f1, f2)
        # r2 = torch.cat([r2_1, r2_2], dim=1) # dim=0横向连接，dim=1纵向连接
        
        # res3
        r3_1 = self.res3_1(r2_1)
        r3_2 = self.res3_2(r2_2)
        r3 = torch.cat([r3_1, r3_2], dim=0)
        
        r3 = self.sk3(r3)
        r3_1, r3_2 = r3[:B], r3[B:]
        
        # res4
        r4_1 = self.res4_1(r3_1)
        r4_2 = self.res4_2(r3_2)
        
        r4 = torch.cat([r4_1, r4_2], dim=0)
        
        r4 = self.sk4(r4)
        r4_1, r4_2 = r4[:B], r4[B:]
        
        
        # res5
        r5_1 = self.res5_1(r4_1)
        r5_2 = self.res5_2(r4_2)
        
        r5 = torch.cat([r5_1, r5_2], dim=0)
        
        r5 = self.sk5(r5)
        r5_1, r5_2 = r5[:B], r5[B:]
        
        return r5_1, [r2_1, r3_1, r4_1], r5_2, [r2_2, r3_2, r4_2]

class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(*[nn.Conv2d(infilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter), nn.ReLU(inplace=True),
                                nn.Conv2d(midfilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter), nn.ReLU(inplace=True),
                                nn.Conv2d(midfilter, outfilter, 3, padding=1, bias=False), nn.BatchNorm2d(outfilter),
                                nn.ReLU(inplace=True)])
    def forward(self, x):
        return self.layer(x)
        
class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(*[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)
    def forward(self, x, H, W):
        x= F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)

class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x

# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.T=4
        
        self.chooseM = chooseMemory(self.T*2,2048)
        
        self.rgb_useM = USEMemory(self.T,2048)
        self.flow_useM = USEMemory(self.T,2048)
        
        self.rgb_updateM = updateMemory(self.T,2048)
        self.flow_updateM = updateMemory(self.T,2048)
       
        self.conv1 = ConvRelu(2048, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256, 256, 3, 1, 1)
        self.cbam1 = CBAM(256)
        self.conv2 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)

        self.cbam2 = CBAM(256)
        self.conv3 = ConvRelu(512, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)

        self.cbam3 = CBAM(256)
        self.conv4 = ConvRelu(256, 256, 1, 1, 0)
        self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)

        self.cbam4 = CBAM(256)

        self.out = out_block(256)

    def forward(self, x, fx, y, fy, rgb_m, flow_m,_class):

        used_m = self.chooseM(rgb_m, flow_m,_class)
        
        x = self.rgb_useM(x,used_m,_class)
        y = self.flow_useM(y,used_m,_class)
        
        new_rgb_m = self.rgb_updateM(rgb_m, x, _class)
        new_flow_m = self.flow_updateM(flow_m, y, _class)
        
        x = self.conv1(x + y)
        x = self.cbam1(self.blend1(x))
        B, C, H, W = fx[2].size()
        feature4 = F.interpolate(x, (H,W), mode='bicubic', align_corners=False)
        x = torch.cat([self.conv2(fx[2] + fy[2]), feature4], dim=1)
        x = self.cbam2(self.blend2(x))
        B, C, H, W = fx[1].size()
        feature3 = F.interpolate(x, (H,W), mode='bicubic', align_corners=False)
        x = torch.cat([self.conv3(fx[1] + fy[1]), feature3], dim=1)
        x = self.cbam3(self.blend3(x))
        B, C, H, W = fx[0].size()
        feature2 = F.interpolate(x, (H,W), mode='bicubic', align_corners=False)
        x = torch.cat([self.conv4(fx[0] + fy[0]), feature2], dim=1)
        x = self.cbam4(self.blend4(x))
        final_score = torch.sigmoid(self.out(x, H * 4, W * 4))
        return final_score, new_rgb_m, new_flow_m
        
if __name__ == '__main__':
    torch.cuda.set_device(device=2)
    input1=torch.tensor([[[1,1],[1,1]],[[1,1],[1,1]]])
    # print(input1[-3::2].shape, input1[-3::2]) # 1; [2]
    print(input1.shape)
    input2 = torch.tensor([0.8,0.2]).reshape((2,1,1))
    print(input1*input2, (input1*input2).shape)
    
    
    
    
from fightingcv_attention.conv.Involution import Involution
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class InvSKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',Involution(kernel_size=k, in_channel=channel)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)

        # fuse
        U=sum(conv_outs)

        # reduction channel
        S=U.mean(-1).mean(-1)
        Z=self.fc(S)

        # calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1))
        attention_weughts=torch.stack(weights,0)
        attention_weughts=self.softmax(attention_weughts)

        # fuse
        V=(attention_weughts*feats).sum(0)
        return V

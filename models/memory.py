import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, T, C):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.T = T
        self.C = C
        self.in_planes = self.T*self.C
        self.fc = nn.Linear(self.in_planes, self.T)

    def forward(self, x, k):

        T, C, H, W = x.shape
        xb = x.view(T*C, H, W)

        f1 = self.fc(self.avg_pool(xb).squeeze().unsqueeze(0))
        f2 = self.fc(self.max_pool(xb).squeeze().unsqueeze(0))

        f = torch.softmax(f1+f2, 1).squeeze()
        indices = torch.argsort(f, dim=-1,descending=True)

        out=torch.index_select(x, dim=0, index=indices)

        return out[:k]

class USEMemory(nn.Module):
    def __init__(self, T, d_model, hv=2, hk=4):
        super(USEMemory, self).__init__()
        self.T=T
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.d_model = d_model
        self.d_v = d_model//hv
        self.d_k = d_model//hk
        self.key = nn.Conv2d(self.d_model, self.d_k, 1)
        self.value = nn.Conv2d(self.d_model, self.d_v, 1)
        self.softmax = nn.Softmax(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.full_connect_1=nn.Linear(2048,1024)
        self.full_connect_2=nn.Linear(1024,2048)

    def forward(self, q, memory, _class):
        
        Memory_buffer = []
        max_num = 1

        for i,idx in enumerate(_class):
            if idx.item() in memory:
                Memory_buffer.append((memory[idx.item()]).unsqueeze(0))
                max_num=max(max_num, memory[idx.item()].shape[0])
            else:
                Memory_buffer.append(q[i].unsqueeze(0).repeat(max_num, 1, 1, 1).unsqueeze(0))
        m = torch.cat(Memory_buffer, dim=0)

        B, C, H, W = q.size()
        Kq, Vq = self.key(q).view(B, H*W, self.d_k),\
                  self.value(q).view(B, H*W, self.d_v)
        B, N, C, H, W = m.size()
        Km, Vm = self.key(m.view(B*N, C, H, W)).view(B, N*H*W, self.d_k),\
                  self.value(m.view(B*N, C, H, W)).view(B, N*H*W, self.d_v)

        K=torch.matmul(Kq,Km.transpose(2,1).contiguous()) # 2,64,384
        K=self.softmax(K)
        Vam=torch.matmul(K,Vm).view(B,H,W,self.d_v) # 2,8,8,1024
        Vq=Vq.view(B,H,W,self.d_v)
        # print(Vam.shape)
        
        gVam=self.gap(Vam.permute(0,3,1,2))
        gVq=self.gap(Vq.permute(0,3,1,2))
        gV=torch.cat((gVam,gVq),dim=1) # 2,2048,1,1
        # print(gV.shape)

        w = torch.sigmoid(self.full_connect_2(self.full_connect_1(gV.view(B,1,1,self.d_model))))
        wm,wq = w[:,:,:,:self.d_v],w[:,:,:,self.d_v:] # 2,1,1,1024
        # print(wm.shape, wq.shape)
        
        Fa=torch.cat((torch.mul(Vam, wm),torch.mul(Vq, wq)),dim=-1).permute(0,3,1,2)
        # print(Fa.shape)
        return Fa

class updateMemory(nn.Module):
    def __init__(self, T, C):
        super(updateMemory, self).__init__()
        self.T=T
        self.TemporalAtt = TemporalAttention(T+1, C)
        
    def forward(self, rgb_m, x, _class):
        new_rgb_m={}
        for i,idx in enumerate(_class):
            if idx.item() in rgb_m:
                new_rgb_m[idx.item()]=self.TemporalAtt(torch.cat((rgb_m[idx.item()],x[i].unsqueeze(0)),dim=0), self.T)
            else:
                new_rgb_m[idx.item()]=x[i].unsqueeze(0).repeat(self.T,1,1,1)
        return new_rgb_m

class chooseMemory(nn.Module):
    def __init__(self, T, C):
        super(chooseMemory, self).__init__()
        self.T=T
        self.TemporalAtt = TemporalAttention(T, C)
        
    def forward(self, rgb_m, flow_m, _class):
        new_rgb_m={}
        for i,idx in enumerate(_class):
            if idx.item() in rgb_m:
                new_rgb_m[idx.item()]=self.TemporalAtt(torch.cat((rgb_m[idx.item()],flow_m[idx.item()]),dim=0), self.T//2)
        return new_rgb_m

if __name__=='__main__':

    torch.cuda.set_device(device=2)
    rtran = updateMemory(5,2048,4).cuda()
    x = torch.randn(4,2048,8,8).cuda()

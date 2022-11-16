import os,sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d,Conv2d
import torch.nn.functional as F
from torch.autograd import Variable

class mlp_conv(nn.Module):
    def __init__(self,in_channels,layer_dim):
        super(mlp_conv,self).__init__()
        self.conv_list=nn.ModuleList()
        for i,num_out_channel in enumerate(layer_dim[:-1]):
            if i==0:
                sub_module=nn.Sequential(
                    Conv1d(in_channels=in_channels, out_channels=num_out_channel, kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
            else:
                sub_module=nn.Sequential(
                    Conv1d(in_channels=layer_dim[i-1],out_channels=num_out_channel,kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
        self.conv_list.append(
            Conv1d(in_channels=layer_dim[-2],out_channels=layer_dim[-1],kernel_size=1)
        )
    def forward(self,inputs):
        net=inputs
        for module in self.conv_list:
            net=module(net)
        return net
class attention_unit(nn.Module):
    def __init__(self,in_channels=130):
        super(attention_unit,self).__init__()
        self.convF=nn.Sequential(
            Conv1d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1),
            nn.ReLU()
        )
        self.convG = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels// 4, kernel_size=1),
            nn.ReLU()
        )
        self.convH = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU()
        )
        # self.gamma=nn.Parameter(torch.tensor(torch.zeros([1])))
        self.gamma=nn.Parameter(torch.zeros([1]).clone().detach())
    def forward(self,inputs):
        f=self.convF(inputs)
        g=self.convG(inputs)#b,32,n
        h=self.convH(inputs)
        s=torch.matmul(g.permute(0,2,1),f)#b,n,n
        beta=F.softmax(s,dim=2)#b,n,n

        o=torch.matmul(h,beta)#b,130,n
        x=self.gamma.to(inputs.device)*o+inputs

        return x
class mlp(nn.Module):
    def __init__(self,in_channels,layer_dim):
        super(mlp,self).__init__()
        self.mlp_list=nn.ModuleList()
        for i,num_outputs in enumerate(layer_dim[:-1]):
            if i==0:
                sub_module=nn.Sequential(
                    nn.Linear(in_channels, num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
            else:
                sub_module=nn.Sequential(
                    nn.Linear(layer_dim[i-1],num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
        self.mlp_list.append(
            nn.Linear(layer_dim[-2],layer_dim[-1])
        )
    def forward(self,inputs):
        net=inputs
        for sub_module in self.mlp_list:
            net=sub_module(net)
        return net



class Discriminator(nn.Module):
    # def __init__(self,params,in_channels):
    def __init__(self,in_channels):
        super(Discriminator,self).__init__()
        # self.params=params
        self.start_number=32
        self.mlp_conv1=mlp_conv(in_channels=in_channels,layer_dim=[self.start_number, self.start_number * 2])
        self.attention_unit=attention_unit(in_channels=self.start_number*4)
        self.mlp_conv2=mlp_conv(in_channels=self.start_number*4,layer_dim=[self.start_number*4,self.start_number*8])
        self.mlp=mlp(in_channels=self.start_number*8,layer_dim=[self.start_number * 8, 1])
    def forward(self,inputs):
        features=self.mlp_conv1(inputs)
        features_global=torch.max(features,dim=2)[0] ##global feature
        features=torch.cat([features,features_global.unsqueeze(2).repeat(1,1,features.shape[2])],dim=1)
        features=self.attention_unit(features)

        features=self.mlp_conv2(features)
        features=torch.max(features,dim=2)[0]

        output=self.mlp(features)

        return output
    # def set_requires_grad(self, nets, requires_grad=False):
    #     if not isinstance(nets, list):
    #         nets = [nets]
    #     for net in nets:
    #         if net is not None:
    #             for param in net.parameters():
    #                 param.requires_grad =  requires_grad
if __name__=="__main__":
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    point_cloud=torch.rand(4,2,500).to(device)
    discriminator=Discriminator(in_channels=2).to(device)
    dis_output=discriminator(point_cloud)
    print(dis_output.shape)
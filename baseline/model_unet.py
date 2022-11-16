import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable



class Generator(nn.Module):
    """G"""

    def __init__(self,dropout_p=0.25):
        super().__init__()
        # encoder input [B x 1 x 20000]
        kernel_size = 10
        
        self.down1 = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size) # [B x 1 x 2000]
        self.enc1_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  # SAME padding [B x 4 x 2000]
        self.enc1_2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)  # SAME padding [B x 4 x 2000]
        self.down2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)  # SAME padding [B x 4 x 500]
        # self.dropout = nn.Dropout(dropout_p)  # feature size:[B x 4 x 500]
        self.enc2_1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.enc2_2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.down3 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.enc3_1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1,padding=1)
        self.enc3_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1,padding=1)
        
        self.dropout = nn.Dropout(dropout_p)
        
        #decoder output [B x 2 x 20000]
        self.up3=nn.ConvTranspose1d(16,8, kernel_size=2, stride=2)
        self.dec3_1=nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.dec3_2=nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.up2=nn.ConvTranspose1d(8, 4, kernel_size=2, stride=2)
        self.dec2_1=nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1)
        self.dec2_2=nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.up1=nn.ConvTranspose1d(4, 2, kernel_size=kernel_size, stride=kernel_size)
        self.dec1_1=nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.dec1_2=nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.final=nn.Conv1d(2, 2, kernel_size=1, stride=1)
        
    def forward(self, spectrum_holder):
        
        e1 = self.down1(spectrum_holder)
        e1 = F.relu(self.enc1_1(e1))
        e1 = F.relu(self.enc1_2(e1))
        # e1 = self.dropout(e1)
        e2 = self.down2(e1)
        e2 = F.relu(self.enc2_1(e2))
        e2 = F.relu(self.enc2_2(e2))
        # e2 = self.dropout(e2)
        e3=self.down3(e2)
        e3=F.relu(self.enc3_1(e3))
        e3=F.relu(self.enc3_2(e3))
        
        d3 = self.up3(e3)
        d3 = torch.cat([e2, d3], 1)
        d3 = F.relu(self.dec3_1(d3))
        d3 = F.relu(self.dec3_2(d3))
        d2 = self.up2(d3)
        d2 = torch.cat([e1, d2], 1)
        d2 = F.relu(self.dec2_1(d2))
        d2 = F.relu(self.dec2_2(d2))
        d1 = self.up1(d2)
        d1 = torch.cat([spectrum_holder, d1], 1)
        d1 = F.relu(self.dec1_1(d1))
        d1 = F.relu(self.dec1_2(d1))
        out = self.final(d1)
        out = F.log_softmax(out.view(-1,2), dim=-1)
        
        return out




if __name__ == '__main__':
    device = torch.device("cuda:0")
    sim_data = Variable(torch.rand(32,1,20000))
    sim_data=sim_data.to(device)
    trans = Generator()
    trans=trans.to(device)
    out = trans(sim_data)
    print(out)
    print('stn', out.size())
    # data=Variable(torch.rand(128,2,2048))
    # data=data.to(device)
    # model=Discriminator()
    # model=model.to(device)
    # ref_x=Variable(torch.rand(128,2,2048))
    # ref_x=ref_x.to(device)
    # output=model(data,ref_x)
    # print(output.shape)
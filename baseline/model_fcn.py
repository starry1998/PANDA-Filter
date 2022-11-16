import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable



#delete z #change decoder input
class Generator(nn.Module):
    """G"""

    def __init__(self,dropout_p=0.25):
        super().__init__()
        # encoder input [B x 1 x 20000]
        kernel_size = 10
        
        self.max_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size , return_indices=True) # [B x 1 x 2000]
        self.enc_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2)  # SAME padding [B x 4 x 2000]
        self.enc_2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2)  # SAME padding [B x 4 x 2000]
        self.max_pool_2 = nn.MaxPool1d(kernel_size=6, stride=4, padding=1, return_indices=True)  # SAME padding [B x 4 x 500]
        self.dropout = nn.Dropout(dropout_p)  # feature size:[B x 4 x 500]
        
        #decoder output [B x 2 x 20000]
        self.dec_1=nn.ConvTranspose1d(4, 2, kernel_size=6, stride=4, padding=1) #[B x 2 x 2000]
        self.bn1=nn.BatchNorm1d(2)
        self.dec_2=nn.ConvTranspose1d(2, 2, kernel_size=7, stride=5, padding=1) #[B x 2 x 10000]
        self.bn2=nn.BatchNorm1d(2)
        self.dec_3=nn.ConvTranspose1d(2, 2, kernel_size=4, stride=2, padding=1) #[B x 2 x 20000]
        self.bn3=nn.BatchNorm1d(2)
        self.classifier=nn.Conv1d(2, 2, kernel_size=1) #[B x 2 x 20000]
        
        
        
    

    def forward(self, spectrum_holder):
        
        net,index1 = self.max_pool_1(spectrum_holder)
        net = F.relu(self.enc_1(net))
        net = F.relu(self.enc_2(net))
        net,index2 = self.max_pool_2(net)
        net = self.dropout(net)
        
        net=self.bn1(F.relu(self.dec_1(net)))
        net=self.bn2(F.relu(self.dec_2(net)))
        net=self.bn3(F.relu(self.dec_3(net)))
        net=self.classifier(net)
        out = F.log_softmax(net.view(-1,2), dim=-1)
        
        return out




if __name__ == '__main__':
    device = torch.device("cuda:0")
    sim_data = Variable(torch.rand(32,1,32000))
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
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.

    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # define gamma and beta parameters
        self.gamma = Parameter(torch.tensor(np.random.normal(1.0, 0.02, size=(1, num_features, 1)),dtype=torch.float32))
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.

        Args:
            x: input tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        Result:
            x: normalized batch tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.

        Args:
            x: input tensor
            mean: mean over features
            mean_sq: squared means over features

        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))

class Generator(nn.Module):
    """G"""

    def __init__(self,dropout_p=0.25):
        super().__init__()
        
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)  # [B x 16 x 1024]
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv1d(16, 32, 4, 2, 1)  # [B x 32 x 512]
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(32, 64, 4, 2, 1)  # [B x 64 x 256]
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(64, 128, 4, 2, 1)  # [B x 128 x 128]
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(128, 512, 4, 2, 1)  # [B x 512 x 64]
        self.enc5_nl = nn.PReLU()
        
        # decoder
        self.dec5 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.dec5_nl = nn.PReLU()  # out : [B x 512 x 128]
        self.dec4 = nn.ConvTranspose1d(512, 256, 4, 2, 1)  # [B x 256 x 256]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(256, 128, 4, 2, 1)  # [B x 128 x 512]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(128, 64, 4, 2, 1)  # [B x 64 x 1024]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 32, 4, 2, 1)  # [B x 32 x 2048]
        self.dec1_nl = nn.PReLU()
        self.classifier = nn.Conv1d(32, 2, kernel_size=1) # [B x 2 x 2048]

    

    def forward(self, spectrum_holder):
        
        # encoding step
        e1 = self.enc1(spectrum_holder)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        
        c = self.enc5_nl(e5)

        # decoding step
        d5 = self.dec5(c)
        d5_c = self.dec5_nl(d5)
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(d4)
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(d3)
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(d2)
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(d1)
        
        d = self.classifier(d1_c)
        out = F.log_softmax(d.view(-1,2), dim=-1)
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
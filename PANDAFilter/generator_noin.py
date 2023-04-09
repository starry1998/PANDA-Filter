from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



class STN3d(nn.Module):
    def __init__(self,k=13):
        super(STN3d, self).__init__()
        #MLP均由卷积结构完成
        #比如说将三维映射到64维。其利用64个1*3的卷积核
        # self.conv1 = torch.nn.Conv1d(2, 64, 1)
        # self.conv1=torch.nn.Conv1d(3, 64, 1)
        self.conv1=torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 4)
        # self.fc3=nn.Linear(256, 9)
        self.fc3=nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        #防止模型过拟合
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # iden = Variable(torch.from_numpy(np.array([1,0,0,1]).astype(np.float32))).view(1,4).repeat(batchsize,1)
        # iden= Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize, 1)
        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        # x = x.view(-1, 2, 2)
        # x = x.view(-1, 3, 3)
        x = x.view(-1, self.k, self.k)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
#特征生成模型
class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, dim=13):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(dim)
        # self.conv1 = torch.nn.Conv1d(2, 64, 1)
        # self.conv1=torch.nn.Conv1d(3, 64, 1)
        self.conv1=torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]#保存点数
        trans = self.stn(x)#STN3 t-net得到变化矩阵
        x = x.transpose(2, 1)#交换tensor的两个维度，便于和旋转矩阵进行计算
        x = torch.bmm(x, trans)#两个batch矩阵乘法，调整姿态
        x = x.transpose(2, 1)#计算完成，转换为原始形式
        x = F.relu(self.bn1(self.conv1(x)))#第一次mlp，每个点由三维->64维
        #是否进行特征转换
        if self.feature_transform:
            trans_feat = self.fstn(x)#得到变换之后的矩阵
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x#保留经过第一次mlp的特征，用于后续与全局特征拼接，用于分割任务
        x = F.relu(self.bn2(self.conv2(x)))#第二次mlp的第一层
        x = self.bn3(self.conv3(x))#第二次mlp的第二层
        x = torch.max(x, 2, keepdim=True)[0]#最大池化作为对称函数
        x = x.view(-1, 1024)#resize池化结果的形状，获得全局的特征
        if self.global_feat:#是否进行全局特征和局部特征的拼接
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)#扩张维度 n_pts为点数n
            return torch.cat([x, pointfeat], 1), trans, trans_feat#cat函数，完成特征的拼接
        
        

    
#分割网络
class PointNetDenseCls(nn.Module):
    def __init__(self, d=13, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform, dim=d)
        #SharedLayer
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        #tower1
        self.tower1 = torch.nn.Conv1d(128, self.k, 1)
        # #tower2
        # self.tower2=torch.nn.Conv1d(128,1,1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]#size()返回张量各个维度的尺寸
        n_pts = x.size()[2]#每个物体的点数
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x1 = self.tower1(x)
        x1 = x1.transpose(2,1).contiguous()
        x1 = F.log_softmax(x1.view(-1,self.k), dim=-1)
        x1 = x1.view(batchsize, n_pts, self.k)
        # x2 = self.tower2(x)
        # x2 = torch.sigmoid(x2)
        # x2=x2.view(batchsize, n_pts, 1)
        # return x1, x2, trans, trans_feat
        return x1,trans,trans_feat
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    # sim_data = Variable(torch.rand(32,13,500))
    # print(sim_data)
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))

    # sim_data_64d = Variable(torch.rand(32, 64, 500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))

    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())
    
    
    # sim_data_cls = Variable(torch.rand(32,4,500))
    # cls = PointNetCls(d = 4,k = 1)
    # out, _, _ = cls(sim_data_cls)
    # print('class', out.size())
    sim_data_seg = Variable(torch.rand(32,13,500))
    seg = PointNetDenseCls(d = 13,k = 2)
    out1, _, _ = seg(sim_data_seg)
    print('m\z', out1.size())
   
#分类网络
# class PointNetCls(nn.Module):
#     def __init__(self, d=2, k=1, feature_transform=False):
#         super(PointNetCls, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, dim=d)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()
# #在执行model(data)时，forward()函数自动调用
#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)#backone
#         x = F.relu(self.bn1(self.fc1(x)))#第三次mlp的第一层
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))#第三次mlp的第二层
#         x = self.fc3(x)#全连接得到k维
#         return torch.sigmoid(x), trans, trans_feat#解决softmax在计算e的次方时容易造成的上溢出和下溢出的问题
    
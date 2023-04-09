import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from generator import PointNetDenseCls,feature_transform_regularizer
from discriminator import Discriminator
from data_reader import ShapeNetDataset
from Focal_Loss import focal_loss
import random
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--classnumber', type=int, help='number of classes', default=2)
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--g_model', type=str, default='/data/lingtianze/zyy/denoise/seg_model/sr015_64_13/generator/generator_model_499.pth', help='generator model path')
parser.add_argument('--d_model', type=str, default='/data/lingtianze/zyy/denoise/seg_model/sr015_64_13/discriminator/discri_model_499.pth', help='discriminator model path')
parser.add_argument('--testdataset', type=str, default='/data/lingtianze/zyy/denoise/data/dataset/testdata_set/inde_testset/PXD010595_test13.pkl', help="dataset path")
parser.add_argument('--feature_transform', default=False, action='store_true', help="use feature transform")
parser.add_argument('--featuresize', type=int, default=13)
opt = parser.parse_args()

# opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed = 2379
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)

test_dataset=ShapeNetDataset(opt.testdataset, dim=opt.featuresize)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=False,
    num_workers=int(opt.workers))


#加载模型
Generator = PointNetDenseCls(d=opt.featuresize , k=opt.classnumber, feature_transform=opt.feature_transform)
discri=Discriminator(in_channels=2)
#加载模型参数
Generator.load_state_dict(torch.load(opt.g_model,map_location='cpu'))
discri.load_state_dict(torch.load(opt.d_model,map_location='cpu'))

device = torch.device("cuda: 0 " if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print('Use multiple GPUs!')
    Generator=nn.DataParallel(Generator)
    discri=nn.DataParallel(discri)
Generator = Generator.to(device)
discri=discri.to(device)
# Generator = Generator.eval()

tp=0
fp=0
tn=0
fn=0
test_loss_g=0
test_loss_d=0
test_intensity_loss=0
with torch.no_grad():
    for j, data in enumerate(testdataloader,0):
        points_noise_feat, points_clean, mz_target, intensity_target = data
        points_noise_feat = points_noise_feat.transpose(2, 1)
        points_clean = points_clean.transpose(2, 1)
        points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
        
        #测试Generator模型的性能，用标签的方法计算损失函数
        mz_pred, intensity_pred, _, _ = Generator(points_noise_feat)
        Generator = Generator.eval()
        mz_pred = mz_pred.view(-1, opt.classnumber)
        mz_pred_=mz_pred
        intensity_pred_=intensity_pred
        mz_target = mz_target.view(-1, 1)[:, 0]
        mz_target=mz_target.long()
        # #不计算padding peak 损失
        
        padding_mask=(mz_target==-100)
        mz_target=mz_target[~padding_mask]
        mz_pred=mz_pred[~padding_mask,:]
        #得到分类的结果
        mz_pred_choice = mz_pred.data.max(1)[1]
        
        #不计算padding的peak预测损失
        intensity_pred=intensity_pred.view(-1, 1)
        intensity_target = intensity_target.view(-1,1)[:, 0]
        intensity_target=intensity_target[~padding_mask]
        intensity_pred=intensity_pred[~padding_mask,:]
        #不计算noise peak 的强度预测损失
        intensity_pred=intensity_pred.view(-1,1)
        mask=(mz_pred_choice==1)
        intensity_pred=intensity_pred[mask,:]
        intensity_target=intensity_target.view(-1, 1)[:, 0]
        intensity_target=intensity_target[mask]
        intensity_target=intensity_target.float()
        intensity_pred=intensity_pred.squeeze()
        #强度预测的损失
        intensity_loss=F.mse_loss(intensity_pred,intensity_target)

        #discriminator 的输出
        mz_pred_choice_ = mz_pred_.data.max(1)[1]
        mz_mask=mz_pred_choice_.unsqueeze(0)
        mz_mask=mz_mask.view(-1,500)
        intensity_mask=mz_mask.unsqueeze(2)
        spectrum_noise_mz=points_noise_feat[:, 0, :]
        #预测得到的m/z
        spectrum_noise_mz=spectrum_noise_mz.masked_fill(mz_mask==0, 0)
        spectrum_noise_mz=spectrum_noise_mz.unsqueeze(2)
        #预测得到的intensity
        intensity_pred_=intensity_pred_.masked_fill(intensity_mask==0, 0)
        #得到预测的谱图
        spectrum_g=torch.cat([spectrum_noise_mz, intensity_pred_],dim=2)
        spectrum_g=spectrum_g.transpose(2,1)
        #discriminator输出的分数
        d_score = discri(spectrum_g)
        discri = discri.eval()
        loss_d=torch.mean((d_score)**2)
        test_loss_d=test_loss_d+loss_d
        #计算评价指标
        tp+=((mz_pred_choice == 1) & (mz_target.data == 1)).cpu().sum()
        tn+=((mz_pred_choice == 0) & (mz_target.data == 0)).cpu().sum()
        fn+=((mz_pred_choice == 0) & (mz_target.data == 1)).cpu().sum()
        fp+=((mz_pred_choice == 1) & (mz_target.data == 0)).cpu().sum()
        rmse=torch.sqrt(intensity_loss)
        test_intensity_loss+=rmse.item()
    p=tp.item() / (fp.item() + tp.item())
    r=tp.item() / (tp.item() + fn.item())
    f1_score=2*r*p/(r+p)
    accuracy=(tp.item()+tn.item())/(tp.item()+tn.item()+fn.item()+fp.item())
    mIOU=1/opt.classnumber*((tp.item() / (fp.item() + fn.item() + tp.item()))+(tn.item() / (fp.item() + fn.item() + tn.item())))
    print('在测试集上的accuracy:{}'.format(accuracy))
    print('在测试集上的F1_score:{}'.format(f1_score))
    print('在测试集上的mIOU:{}'.format(mIOU))
    print('在测试集上的recall:{}'.format(r))
    print('在测试集上的precision:{}'.format(p))
    print('在测试集上的rmse:{}'.format(test_intensity_loss/j))
    print('在测试集上的loss_d:{}'.format(test_loss_d/j))
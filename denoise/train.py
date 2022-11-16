from __future__ import print_function
import argparse
from functools import total_ordering
import random
import  os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data_reader import ShapeNetDataset
from model import PointNetDenseCls,feature_transform_regularizer
from PU_GAN_Discriminator import Discriminator
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from Focal_Loss import focal_loss
import logging
from datetime import datetime
import sys

#建立日志文档
def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./logs/',
                only_file=False):
   # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                           format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                           )
#获取参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=128, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--classnumber', type=int, help='number of classes', default=2)
parser.add_argument(
    '--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--traindataset', type=str, default='/data/lingtianze/zyy/denoise/data/dataset/traindata_set/traindata_nce25_sn01.pkl', help="dataset path")
parser.add_argument('--testdataset', type=str, default='/data/lingtianze/zyy/denoise/data/dataset/traindata_set/PXD010595_25.pkl', help="dataset path")
parser.add_argument('--feature_transform', default=False, action='store_true', help="use feature transform")

opt = parser.parse_args()

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
#将模型参数写入到log
logger_init("test_log",log_level=logging.INFO)
logging.info("-------------loading parameters----------")
logging.info(opt)

#构建数据集
dataset = ShapeNetDataset(opt.traindataset)
# test_dataset_unique= ShapeNetDataset(opt.testdataset)
#划分数据集
train_size=int(len(dataset)*0.8)
val_size=int(len(dataset)*0.1)
test_size=len(dataset)-train_size-val_size
train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
# train_dataset, val_dataset=torch.utils.data.random_split(dataset,[train_size,val_size])
# val_size=int(len(t_dataset))-train_size
# train_dataset, val_dataset =torch.utils.data.random_split(t_dataset,[train_size,val_size])
# test_dataset=ShapeNetDataset(opt.testdataset)
#将数据集的大小输出到log中
logging.info("the size of training_dataset:{},test_dataset:{}".format(len(train_dataset),len(test_dataset)))
#加载数据集
traindataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
    num_workers=int(opt.workers))
valdataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
    num_workers=int(opt.workers))
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=False,
    num_workers=int(opt.workers))


# blue = lambda x: '\033[94m' + x + '\033[0m'
#计算loss权重的函数
def calculate_weigths_labels(mz_target,num_classes):
    mask=(mz_target==0)
    count_l=torch.bincount(~mask, minlength=num_classes)
    total_frequency=torch.sum(count_l)
    class_weights=[]
    for frequency in count_l:
        class_weight=(1/num_classes)/(frequency.item()/total_frequency.item())
        class_weights.append(class_weight)
    signal_noise_ratio=count_l[0]/count_l[1]
    return class_weights

#创建网络模型
#创建生成器
Generator = PointNetDenseCls(d=13 , k=opt.classnumber, feature_transform=opt.feature_transform)
#创建判别器
discri=Discriminator(in_channels=2)
#定义优化器
#定义生成器的优化器
g_optimizer = optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.5)

#定义判别器的优化器
d_optimizer = optim.Adam(discri.parameters(), lr=0.0002, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=20, gamma=0.5)

#设置GPU
#指定GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print('可以使用', torch.cuda.device_count(), 'GPUs!')
    Generator=nn.DataParallel(Generator)
    discri=nn.DataParallel(discri)
Generator=Generator.to(device)
discri=discri.to(device)
train_num_batch = len(train_dataset) / opt.batchSize
val_num_batch=len(val_dataset) / opt.batchSize

#添加tensorboard
writer=SummaryWriter("./logs_train")

for epoch in range(opt.nepoch):
    logging.info("————————模型第{}轮训练开始————————".format(epoch+1))
    scheduler.step()
    epoch_loss_g=0
    epoch_loss_d=0
    epoch_loss_g1=0
    epoch_loss_g2=0
    epoch_loss_d1=0
    epoch_loss_d2=0
    train_rmse=0
    tp=0
    fp=0
    tn=0
    fn=0
    #训练步骤开始
    for i, data in enumerate(traindataloader, 0):
        #读取数据
        points_noise_feat, points_clean, mz_target, intensity_target = data
        points_noise_feat = points_noise_feat.transpose(2, 1)
        points_clean = points_clean.transpose(2, 1)
        points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
       
        #先训练D
        #通过真实理论谱图+实验谱图进行训练
        #真实的理论谱图输出的值越接近1越好
        d_optimizer.zero_grad()
        discri = discri.train()
        d_out01 = discri(points_clean[:,0:2,:])
        clean_loss = torch.mean((d_out01-1)**2)
        epoch_loss_d1=epoch_loss_d1+clean_loss
        
        #再根据G的生成理论谱图进行训练
        mz_pred, intensity_pred, trans, trans_feat = Generator(points_noise_feat)
        mz_pred=mz_pred.view(-1, opt.classnumber)
        mz_pred_choice = mz_pred.data.max(1)[1] #得到分类的结果
        mz_mask=mz_pred_choice.unsqueeze(0) #根据分类的结果作为mz_mask
        mz_mask=mz_mask.view(-1,500) #mz_mask的维度[64,500]
        intensity_mask=mz_mask.unsqueeze(2) #intensity_mask的维度[64,500,1]
        spectrum_noise_mz=points_noise_feat[:, 0, :] #实验谱图的mz
        #预测得到的m/z
        spectrum_noise_mz=spectrum_noise_mz.masked_fill(mz_mask==0, 0)
        spectrum_noise_mz=spectrum_noise_mz.unsqueeze(2)
        #预测得到的intensity
        intensity_pred=intensity_pred.masked_fill(intensity_mask==0, 0)
        #G生成的谱图
        spectrum_g=torch.cat([spectrum_noise_mz, intensity_pred],dim=2)
        spectrum_g=spectrum_g.transpose(2,1)
    
        # d_out2, trans, trans_feat = Discriminator(torch.cat((points_noise[:,0:2,:], spectrum_g),dim=1))
        d_out2 = discri(spectrum_g)
        noisy_loss = torch.mean(d_out2**2) 
        epoch_loss_d2=epoch_loss_d2+noisy_loss
        #将上面两部分的损失加起来计算D的loss
        loss_d=clean_loss + noisy_loss
        epoch_loss_d+=loss_d
        loss_d.backward()
        d_optimizer.step()
        #训练Generator
        #监督学习
        # 峰分类任务
        g_optimizer.zero_grad()
        Generator = Generator.train()
        
        mz_pred, intensity_pred, trans, trans_feat = Generator(points_noise_feat)
        mz_pred=mz_pred.view(-1, opt.classnumber)
        mz_pred_=mz_pred
        intensity_pred_=intensity_pred
        
        #分类的标签
        mz_target = mz_target.view(-1,1)[:, 0]
        mz_target=mz_target.long()
        #不计算padding的peak
        padding_mask=(mz_target==-100)
        mz_target=mz_target[~padding_mask]
        mz_pred=mz_pred[~padding_mask,:]
        #得到分类结果
        mz_pred_choice = mz_pred.data.max(1)[1]
        #计算不同类的权重 #计算峰分类损失
        classes_weights=calculate_weigths_labels(mz_target,opt.classnumber)
        loss_fn = focal_loss(alpha=classes_weights, gamma=2, num_classes=opt.classnumber)
        mz_loss=loss_fn(mz_pred, mz_target)
        
        #强度预测任务
        #不计算padding的peak预测损失
        intensity_pred=intensity_pred.view(-1, 1)
        intensity_target = intensity_target.view(-1,1)[:, 0]
        intensity_target=intensity_target[~padding_mask]
        intensity_pred=intensity_pred[~padding_mask,:]
        #根据mz_pred预测结果进行mask，mask掉预测结果为0的峰的强度预测值
        noise_mask=(mz_pred_choice==1)
        intensity_pred=intensity_pred[noise_mask,:]
        intensity_target=intensity_target[noise_mask]
        intensity_target=intensity_target.float()
        intensity_pred=intensity_pred.squeeze()
        #计算强度预测损失
        intensity_loss=F.mse_loss(intensity_pred, intensity_target)
        #结合两个任务的损失
        loss_g1 = mz_loss + intensity_loss
        epoch_loss_g1+=loss_g1
        
        #训练G#非监督学习 使得D识别G(y)为真
        #mask掉noise peak
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
        # d_out3, trans, trans_feat = Discriminator(torch.cat((points_noise[:,0:2,:], spectrum_g),dim=1))
        d_out3 = discri(spectrum_g)
        loss_g2=torch.mean((d_out3-1)**2)
        epoch_loss_g2+=loss_g2
        
        loss_g= (loss_g1 + loss_g2)/2.0
        epoch_loss_g+=loss_g
        loss_g.backward()
        g_optimizer.step()
        logging.info('[%d: %d/%d] Discriminator loss: %f, Generator loss: %f' % (epoch, i, train_num_batch, loss_d.item(), loss_g.item()))
        
        #计算评价指标
        tp+=((mz_pred_choice == 1) & (mz_target.data == 1)).cpu().sum()
        tn+=((mz_pred_choice == 0) & (mz_target.data == 0)).cpu().sum()
        fn+=((mz_pred_choice == 0) & (mz_target.data == 1)).cpu().sum()
        fp+=((mz_pred_choice == 1) & (mz_target.data == 0)).cpu().sum()
        rmse = torch.sqrt(intensity_loss)
        train_rmse+=rmse
    p=tp.item()/float(tp.item()+fp.item())
    r=tp.item()/float(tp.item()+fn.item())
    f1_score=2*r*p/(r+p)
    accuracy=(tp.item()+tn.item())/(tp.item()+tn.item()+fn.item()+fp.item())
    mIOU=1/opt.classnumber*((tp.item() / (fp.item() + fn.item() + tp.item()))+(tn.item() / (fp.item() + fn.item() + tn.item())))
    logging.info('在训练集上的accuracy:{},在训练集上的F1_score:{},在训练集上的mIOU:{}'.format(accuracy,f1_score,mIOU))
    writer.add_scalar("训练集上的生成器的损失", epoch_loss_g, epoch)
    writer.add_scalar("训练集上的判别器的损失", epoch_loss_d, epoch)
    writer.add_scalar("mIOU", mIOU, epoch)
    writer.add_scalar("f1_score", f1_score, epoch)
    # writer.add_scalar("训练集上的生成器-分类器的损失", epoch_loss_g1, epoch)
    # writer.add_scalar("训练集上的生成器-判别器的损失", epoch_loss_g2, epoch)
    # writer.add_scalar("训练集上的判别器-理论谱图的损失", epoch_loss_d1, epoch)
    # writer.add_scalar("训练集上的判别器-生成谱图的损失", epoch_loss_d1, epoch)
    # 验证步骤开始
  
    tp=0
    fp=0
    tn=0
    fn=0
    logging.info("——————————模型验证————————————")    
    val_loss_g=0
    val_loss_d=0
    val_intensity_loss=0
    with torch.no_grad():
        for j, data in enumerate(valdataloader,0):
            points_noise_feat, points_clean, mz_target, intensity_target = data
            points_noise_feat = points_noise_feat.transpose(2, 1)
            points_clean = points_clean.transpose(2, 1)
            points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
            Generator = Generator.eval()
            #测试Generator模型的性能，用标签的方法计算损失函数
            mz_pred, intensity_pred, _, _ = Generator(points_noise_feat)
            mz_pred = mz_pred.view(-1, opt.classnumber)
            mz_pred_=mz_pred
            intensity_pred_=intensity_pred
            mz_target = mz_target.view(-1, 1)[:, 0]
            mz_target=mz_target.long()
            #不计算padding peak 损失
            
            padding_mask=(mz_target==-100)
            mz_target=mz_target[~padding_mask]
            mz_pred=mz_pred[~padding_mask,:]
            #得到分类的结果
            mz_pred_choice = mz_pred.data.max(1)[1]
            #峰分类的损失
            loss_fn = focal_loss(alpha=0.25, gamma=3, num_classes=2)
            mz_loss=loss_fn(mz_pred, mz_target)
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
            #损失的结合
            loss=(mz_loss+intensity_loss)/2
            #验证集整体的损失
            val_loss_g=val_loss_g+loss
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
            # d_score, trans, trans_feat = Discriminator(torch.cat((points_noise[:,0:2,:], spectrum_g),dim=1))
            d_score = discri(spectrum_g)
            loss_d=torch.mean((d_score-1)**2)
            #测试集Discriminator的损失
            val_loss_d=val_loss_d+loss_d
            
            # 计算准确度
            tp+=((mz_pred_choice == 1) & (mz_target.data == 1)).cpu().sum()
            tn+=((mz_pred_choice == 0) & (mz_target.data == 0)).cpu().sum()
            fn+=((mz_pred_choice == 0) & (mz_target.data == 1)).cpu().sum()
            fp+=((mz_pred_choice == 1) & (mz_target.data == 0)).cpu().sum()
            rmse=torch.sqrt(intensity_loss)
            val_intensity_loss+=rmse.item()
        p=tp.item() / (fp.item() + tp.item())
        r=tp.item() / (tp.item() + fn.item())
        f1_score=2*r*p/(r+p)
        accuracy=(tp.item()+tn.item())/(tp.item()+tn.item()+fn.item()+fp.item())
        mIOU=1/opt.classnumber*((tp.item() / (fp.item() + fn.item() + tp.item()))+(tn.item() / (fp.item() + fn.item() + tn.item())))
        logging.info('在验证集上的accuracy:{}'.format(accuracy))
        logging.info('在验证集上的F1_score:{}'.format(f1_score))
        logging.info('在验证集上的mIOU:{}'.format(mIOU))
        logging.info('在验证集上的recall:{}'.format(r))
        logging.info('在验证集上的precision:{}'.format(p))
        logging.info('在验证集上的rme:{}'.format(val_intensity_loss/j))
        logging.info('在验证集上的loss_d:{}'.format(val_loss_d/j))
        writer.add_scalar("验证集上的生成器的损失", val_loss_g, epoch)
        writer.add_scalar("验证集上的判别器的损失", val_loss_d, epoch)
 
    torch.save(Generator.module.state_dict(), '%s/%s/generator_model_%d.pth' % ('seg_model', 'generator',epoch))
    torch.save(discri.module.state_dict(), '%s/%s/discri_model_%d.pth' % ('seg_model', 'discri',epoch))

# 测试Generator的预测效果，以及Discriminator的判断结果
logging.info("———————————————模型测试————————————")
test_loss_g=0
test_loss_d=0
test_intensity_loss=0
tp=0
fp=0
tn=0
fn=0
with torch.no_grad():
    for j, data in enumerate(testdataloader,0):
        points_noise_feat, points_clean, mz_target, intensity_target = data
        points_noise_feat = points_noise_feat.transpose(2, 1)
        points_clean = points_clean.transpose(2, 1)
        points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
        Generator = Generator.eval()
        #测试Generator模型的性能，用标签的方法计算损失函数
        mz_pred, intensity_pred, _, _ = Generator(points_noise_feat)
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
        loss_d=torch.mean((d_score-1)**2)
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
    logging.info('在测试集上的accuracy:{}'.format(accuracy))
    logging.info('在测试集上的F1_score:{}'.format(f1_score))
    logging.info('在测试集上的mIOU:{}'.format(mIOU))
    logging.info('在测试集上的recall:{}'.format(r))
    logging.info('在测试集上的precision:{}'.format(p))
    logging.info('在测试集上的rmse:{}'.format(test_intensity_loss/j))
    logging.info('在测试集上的loss_d:{}'.format(test_loss_d/j))
# model_parameters={}
# parameters_dict=Generator.module.state_dict()
# for name,parameters in Generator.named_parameters():
#     model_parameters[name]=parameters
#     tensor1=parameters
#     tensor2=parameters_dict[name]
#     print(tensor1.equal(tensor2))
# print(model_parameters.keys())  #len=50
# print(parameters_dict.keys())  #len=83
# np.save('/data/lingtianze/zyy/denoise/seg_model/cache/parameters.npy', model_parameters)
# torch.save(Generator, '%s/%s/Generator_1.pth' % ('seg_model','cache'))
# torch.save(discri, '%s/%s/discri_1.pth' % ('seg_model','cache'))

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
from generator import PointNetDenseCls,feature_transform_regularizer
from discriminator import Discriminator
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from Focal_Loss import focal_loss
import logging
from datetime import datetime
import sys

#建立日志文档
def logger_init(log_file_name='monitor',log_level=logging.DEBUG,log_dir='./logs/',only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + '.txt')
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
    '--batchsize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--classnumber', type=int, help='number of classes', default=2)
parser.add_argument(
    '--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--traindataset', type=str, default='/data/lingtianze/zyy/denoise/data/dataset/traindata_set/trainset_sample40k.pkl', help="dataset path")
parser.add_argument('--validataset', type=str, default='/data/lingtianze/zyy/denoise/data/dataset/valdata_set/val_lr08.pkl', help="dataset path")
parser.add_argument('--testdataset', type=str, default='/data/lingtianze/zyy/denoise/data/dataset/testdata_set/test_lr08.pkl', help="dataset path")
parser.add_argument('--feature_transform', default=False, action='store_true', help="use feature transform")
parser.add_argument('--log_filename', type=str, default='lr0001_64_500_NF_13')
parser.add_argument('--featuresize', type=int, default=2)
opt = parser.parse_args()

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
#log file
logger_init(opt.log_filename,log_level=logging.INFO)
logging.info("-------------loading parameters----------")
logging.info(opt)
#tensorboard
tensor_path="./logs_train/" + opt.log_filename
if not os.path.exists(tensor_path):
    os.makedirs(tensor_path)
writer=SummaryWriter(tensor_path)

#traindataset,valdataset,testdataset
train_dataset = ShapeNetDataset(opt.traindataset,dim=opt.featuresize)
val_dataset = ShapeNetDataset(opt.validataset,dim=opt.featuresize)
test_dataset= ShapeNetDataset(opt.testdataset,dim=opt.featuresize)
#构建数据集
# dataset = ShapeNetDataset(opt.traindataset,dim=opt.featuresize)
#划分数据集
# train_size=int(len(dataset)*0.8)
# val_size=int(len(dataset)*0.1)
# test_size=len(dataset)-train_size-val_size
# train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])

#the size of dataset
logging.info("the size of training_dataset:{},val_dataset:{}".format(len(train_dataset),len(val_dataset)))
#dataloader
traindataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=True,
    num_workers=int(opt.workers))
valdataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=True,
    num_workers=int(opt.workers))
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=False,
    num_workers=int(opt.workers))
# #regularization
# def L2Loss(model,alpha):
#     l2_loss=torch.tensor(0.0,requires_grad=True)
#     for name, parma in model.named_parameters():
#         if 'bias' not in name:
#             l2_loss=l2_loss+(0.5*alpha*torch.sum(torch.pow(parma,2)))
#     return l2_loss


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

#build model
#optimizer
#segmention model
Generator = PointNetDenseCls(d=opt.featuresize, k=opt.classnumber, feature_transform=opt.feature_transform)

g_optimizer = optim.RMSprop(Generator.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.5)
#discriminator model
discri=Discriminator(in_channels=2)

d_optimizer = optim.RMSprop(discri.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=20, gamma=0.5)

#use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print('can use', torch.cuda.device_count(), 'GPUs!')
    Generator=nn.DataParallel(Generator)
    discri=nn.DataParallel(discri)
Generator=Generator.to(device)
discri=discri.to(device)
train_num_batch = len(train_dataset) / opt.batchsize
val_num_batch=len(val_dataset) / opt.batchsize


for epoch in range(opt.nepoch):
    logging.info("————————the training of {} eopch————————".format(epoch+1))
    scheduler.step()
    loss_train_g=0
    loss_train_d=0
    loss_train_g_seg=0
    loss_train_g_adv=0
    loss_train_mse=0
    tp_train=0
    fp_train=0
    tn_train=0
    fn_train=0
    tp_vali=0
    fp_vali=0
    tn_vali=0
    fn_vali=0
    for i, data in enumerate(traindataloader, 0):
        #read data
        points_noise_feat, points_clean, mz_target, intensity_target = data
        points_noise_feat = points_noise_feat.transpose(2, 1)
        points_clean = points_clean.transpose(2, 1)
        points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
       
        #train discriminator
        #Loss of theoretical spectrum
        d_optimizer.zero_grad()
        discri = discri.train()
        d_out01 = discri(points_clean[:,0:2,:])
        theo_loss = torch.mean((d_out01)**2)
        
        #loss of spectrum from segmentation model
        mz_pred, intensity_pred, trans, trans_feat = Generator(points_noise_feat)
        mz_pred=mz_pred.view(-1, opt.classnumber)
        mz_pred_choice = mz_pred.data.max(1)[1] 
        mz_mask=mz_pred_choice.unsqueeze(0) 
        mz_mask=mz_mask.view(-1,500) 
        intensity_mask=mz_mask.unsqueeze(2) 
        spectrum_noise_mz=points_noise_feat[:, 0, :] 
        #the m/z of prediction
        spectrum_noise_mz=spectrum_noise_mz.masked_fill(mz_mask==0, 0)
        spectrum_noise_mz=spectrum_noise_mz.unsqueeze(2)
        #the intensity of prediction
        intensity_pred=intensity_pred.masked_fill(intensity_mask==0, 0)
        #the spectrum of segmentation
        spectrum_g=torch.cat([spectrum_noise_mz, intensity_pred],dim=2)
        spectrum_g=spectrum_g.transpose(2,1)
        d_out2 = discri(spectrum_g)
        seg_loss = torch.mean((d_out2-1)**2) 

        #the loss of discriminator
        loss_d = theo_loss + seg_loss
        loss_train_d = loss_train_d + loss_d.item()
        loss_d.backward()
        d_optimizer.step()

        #train segmentation model
        #peak class
        g_optimizer.zero_grad()
        Generator = Generator.train()
        
        mz_pred, intensity_pred, trans, trans_feat = Generator(points_noise_feat)
        mz_pred=mz_pred.view(-1, opt.classnumber)
        mz_pred_all=mz_pred
        intensity_pred_all=intensity_pred
        mz_target = mz_target.view(-1,1)[:, 0]
        mz_target=mz_target.long()
        #mask padding peak
        padding_mask=(mz_target==-100)
        mz_target=mz_target[~padding_mask]
        mz_pred=mz_pred[~padding_mask,:]
        #the result of class
        mz_pred_choice = mz_pred.data.max(1)[1]
        #loss of class
        classes_weights=calculate_weigths_labels(mz_target,opt.classnumber)
        loss_fn = focal_loss(alpha=classes_weights, gamma=2, num_classes=opt.classnumber)
        mz_loss=loss_fn(mz_pred, mz_target)
        
        #intensity prediction
        #the loss of segmentation
        #mask padding peak
        intensity_pred=intensity_pred.view(-1, 1)
        intensity_target = intensity_target.view(-1,1)[:, 0]
        intensity_target=intensity_target[~padding_mask]
        intensity_pred=intensity_pred[~padding_mask,:]
        #mask noise peak
        noise_mask=(mz_pred_choice==1)
        intensity_pred=intensity_pred[noise_mask,:]
        intensity_target=intensity_target[noise_mask]
        intensity_target=intensity_target.float()
        intensity_pred=intensity_pred.squeeze()
        #the loss of intensity prediction
        intensity_loss=F.mse_loss(intensity_pred, intensity_target)
        loss_seg = mz_loss + intensity_loss
        loss_train_g_seg=loss_train_g_seg+loss_seg.item()
        #loss of adversarial learning
        #mask noise peak
        mz_pred_choice_all = mz_pred_all.data.max(1)[1]
        mz_mask=mz_pred_choice_all.unsqueeze(0)
        mz_mask=mz_mask.view(-1,500)
        intensity_mask=mz_mask.unsqueeze(2)
        spectrum_noise_mz=points_noise_feat[:, 0, :]
        #the mz of prediction
        spectrum_noise_mz=spectrum_noise_mz.masked_fill(mz_mask==0, 0)
        spectrum_noise_mz=spectrum_noise_mz.unsqueeze(2)
        #the intensity of prediction
        intensity_pred_all=intensity_pred_all.masked_fill(intensity_mask==0, 0)
        #the spectrum of segmentation
        spectrum_g=torch.cat([spectrum_noise_mz, intensity_pred_all],dim=2)
        spectrum_g=spectrum_g.transpose(2,1)
        d_out3 = discri(spectrum_g)
        loss_adv=torch.mean(d_out3**2)
        loss_train_g_adv=loss_train_g_adv+loss_adv.item()
        loss_g= 0.5*loss_seg + 0.5*loss_adv
        loss_train_g=loss_train_g+loss_g.item()
        loss_g.backward()
        g_optimizer.step()

        logging.info('[%d: %d/%d] Discriminator loss: %f, Generator loss: %f' % (epoch, i, train_num_batch, loss_d.item(), loss_g.item()))
        
        #evaluation metric
        tp_train+=((mz_pred_choice == 1) & (mz_target.data == 1)).cpu().sum()
        tn_train+=((mz_pred_choice == 0) & (mz_target.data == 0)).cpu().sum()
        fn_train+=((mz_pred_choice == 0) & (mz_target.data == 1)).cpu().sum()
        fp_train+=((mz_pred_choice == 1) & (mz_target.data == 0)).cpu().sum()
        loss_train_mse+=intensity_loss.item()
    p=tp_train.item()/float(tp_train.item()+fp_train.item())
    r=tp_train.item()/float(tp_train.item()+fn_train.item())
    f1_score=2*r*p/(r+p)
    mIOU=1/opt.classnumber*((tp_train.item() / (fp_train.item() + fn_train.item() + tp_train.item()))+(tn_train.item() / (fp_train.item() + fn_train.item() + tn_train.item())))
    logging.info('the precision of training :{},the recall of training :{},the F1-score of training :{}, the MIoU of training :{}'.format(p,r,f1_score,mIOU))
    logging.info('the loss of intensity prediction :{}'.format(loss_train_mse/i))
    writer.add_scalar("loss_gener", loss_train_g/i, epoch)
    writer.add_scalar("loss_discr", loss_train_d/i, epoch)
    writer.add_scalar("loss_seg", loss_train_g_seg/i, epoch)
    writer.add_scalar("loss_adv", loss_train_g_adv/i, epoch)
    writer.add_scalars("train_performance", {"recall":r, "precision":p, "f1_score":f1_score, "mIoU":mIOU}, epoch)
    # 验证步骤开始
    logging.info("——————————the validation of model————————————")    
    val_loss_g=0
    val_loss_intensity=0
    val_loss_mz=0
    with torch.no_grad():
        for j, data in enumerate(valdataloader,0):
            points_noise_feat, points_clean, mz_target, intensity_target = data
            points_noise_feat = points_noise_feat.transpose(2, 1)
            points_clean = points_clean.transpose(2, 1)
            points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
            Generator = Generator.eval()
            
            mz_pred, intensity_pred, _, _ = Generator(points_noise_feat)
            mz_pred = mz_pred.view(-1, opt.classnumber)
            mz_pred_all=mz_pred
            intensity_pred_all=intensity_pred
            mz_target = mz_target.view(-1, 1)[:, 0]
            mz_target=mz_target.long()
            #mask padding peak
            padding_mask=(mz_target==-100)
            mz_target=mz_target[~padding_mask]
            mz_pred=mz_pred[~padding_mask,:]
            
            mz_pred_choice = mz_pred.data.max(1)[1]
            #the loss of mz
            classes_weights=calculate_weigths_labels(mz_target,opt.classnumber)
            loss_fn = focal_loss(alpha=classes_weights, gamma=2, num_classes=opt.classnumber)
            mz_loss=loss_fn(mz_pred, mz_target)
        
            #mask padding peak
            intensity_pred=intensity_pred.view(-1, 1)
            intensity_target = intensity_target.view(-1,1)[:, 0]
            intensity_target=intensity_target[~padding_mask]
            intensity_pred=intensity_pred[~padding_mask,:]
            #mask noise peak
            intensity_pred=intensity_pred.view(-1,1)
            mask=(mz_pred_choice==1)
            intensity_pred=intensity_pred[mask,:]
            intensity_target=intensity_target.view(-1, 1)[:, 0]
            intensity_target=intensity_target[mask]
            intensity_target=intensity_target.float()
            intensity_pred=intensity_pred.squeeze()
            #the loss of intensity
            intensity_loss=F.mse_loss(intensity_pred,intensity_target)
            #loss of generator
            loss_seg=mz_loss+intensity_loss
           
            
            # discriminator
            mz_pred_choice_all = mz_pred_all.data.max(1)[1]
            mz_mask=mz_pred_choice_all.unsqueeze(0)
            mz_mask=mz_mask.view(-1,500)
            intensity_mask=mz_mask.unsqueeze(2)
            spectrum_noise_mz=points_noise_feat[:, 0, :]
            #mz
            spectrum_noise_mz=spectrum_noise_mz.masked_fill(mz_mask==0, 0)
            spectrum_noise_mz=spectrum_noise_mz.unsqueeze(2)
            #intensity
            intensity_pred_all=intensity_pred_all.masked_fill(intensity_mask==0, 0)
            #spectrum
            spectrum_g=torch.cat([spectrum_noise_mz, intensity_pred_all],dim=2)
            spectrum_g=spectrum_g.transpose(2,1)
           
            d_score = discri(spectrum_g)
            loss_adv=torch.mean((d_score-1)**2)
            loss_g=0.5*loss_seg + 0.5*loss_adv
            
            #the loss of epoch
            val_loss_g=val_loss_g+loss_g.item()
            val_loss_mz=val_loss_mz+mz_loss.item()
            val_loss_intensity=val_loss_intensity+intensity_loss.item()
            # the evaluation metrics
            tp_vali+=((mz_pred_choice == 1) & (mz_target.data == 1)).cpu().sum()
            tn_vali+=((mz_pred_choice == 0) & (mz_target.data == 0)).cpu().sum()
            fn_vali+=((mz_pred_choice == 0) & (mz_target.data == 1)).cpu().sum()
            fp_vali+=((mz_pred_choice == 1) & (mz_target.data == 0)).cpu().sum()
        p=tp_vali.item() / (fp_vali.item() + tp_vali.item())
        r=tp_vali.item() / (tp_vali.item() + fn_vali.item())
        f1_score=2*r*p/(r+p)
        mIOU=1/opt.classnumber*((tp_vali.item() / (fp_vali.item() + fn_vali.item() + tp_vali.item()))+(tn_vali.item() / (fp_vali.item() + fn_vali.item() + tn_vali.item())))
        logging.info('the precision of validation :{},the recall of validation :{},the F1-score of validation :{}, the MIoU of validation :{}'.format(p,r,f1_score,mIOU))
        logging.info('the loss of generator :{}'.format(val_loss_g/j))
        logging.info('the loss of intensity prediction :{}'.format(val_loss_intensity/j))
        logging.info('the loss of mz class :{}'.format(val_loss_mz/j))
        writer.add_scalar("loss_val_mz", val_loss_mz/j, epoch)
        writer.add_scalars("val_performance", {"recall":r, "precision":p, "f1_score":f1_score, "mIoU":mIOU}, epoch)
        
        
    Generator_path=os.path.join('seg_model', opt.log_filename , 'generator')
    Discriminator_path=os.path.join('seg_model', opt.log_filename, 'discriminator')
    if not os.path.exists(Generator_path):
        os.makedirs(Generator_path)
    if not os.path.exists(Discriminator_path):
        os.makedirs(Discriminator_path)
    torch.save(Generator.module.state_dict(), '%s/%s/%s/generator_model_%d.pth' % ('seg_model', opt.log_filename , 'generator', epoch))
    torch.save(discri.module.state_dict(), '%s/%s/%s/discri_model_%d.pth' % ('seg_model', opt.log_filename, 'discriminator',epoch))
    if ((epoch+1) % 20 == 0) :
        logging.info("———————————————the test of model———————————")
        test_loss_intensity=0
        tp_test=0
        fp_test=0
        tn_test=0
        fn_test=0
        with torch.no_grad():
            for f, data in enumerate(testdataloader,0):
                points_noise_feat, points_clean, mz_target, intensity_target = data
                points_noise_feat = points_noise_feat.transpose(2, 1)
                points_clean = points_clean.transpose(2, 1)
                points_noise_feat, points_clean, mz_target, intensity_target = points_noise_feat.to(device), points_clean.to(device), mz_target.to(device), intensity_target.to(device)
                Generator = Generator.eval()

                mz_pred, intensity_pred, _, _ = Generator(points_noise_feat)
                mz_pred = mz_pred.view(-1, opt.classnumber)
                mz_target = mz_target.view(-1, 1)[:, 0]
                mz_target=mz_target.long()
                padding_mask=(mz_target==-100)
                mz_target=mz_target[~padding_mask]
                mz_pred=mz_pred[~padding_mask,:]
                mz_pred_choice = mz_pred.data.max(1)[1]
                
                intensity_pred=intensity_pred.view(-1, 1)
                intensity_target = intensity_target.view(-1,1)[:, 0]
                intensity_target=intensity_target[~padding_mask]
                intensity_pred=intensity_pred[~padding_mask,:]
                intensity_pred=intensity_pred.view(-1,1)
                mask=(mz_pred_choice==1)
                intensity_pred=intensity_pred[mask,:]
                intensity_target=intensity_target.view(-1, 1)[:, 0]
                intensity_target=intensity_target[mask]
                intensity_target=intensity_target.float()
                intensity_pred=intensity_pred.squeeze()
                intensity_loss=F.mse_loss(intensity_pred,intensity_target)
                test_loss_intensity=test_loss_intensity+intensity_loss
                
                
                tp_test+=((mz_pred_choice == 1) & (mz_target.data == 1)).cpu().sum()
                tn_test+=((mz_pred_choice == 0) & (mz_target.data == 0)).cpu().sum()
                fn_test+=((mz_pred_choice == 0) & (mz_target.data == 1)).cpu().sum()
                fp_test+=((mz_pred_choice == 1) & (mz_target.data == 0)).cpu().sum()
            
            p=tp_test.item() / (fp_test.item() + tp_test.item())
            r=tp_test.item() / (tp_test.item() + fn_test.item())
            f1_score=2*r*p/(r+p)
            mIOU=1/opt.classnumber*((tp_test.item() / (fp_test.item() + fn_test.item() + tp_test.item()))+(tn_test.item() / (fp_test.item() + fn_test.item() + tn_test.item())))
            
            logging.info('the recall of test:{}'.format(r))
            logging.info('the precision of test:{}'.format(p))
            logging.info('the F1_score of test:{}'.format(f1_score))
            logging.info('the mIOU of test:{}'.format(mIOU))
            logging.info('the loss of intensity prediction:{}'.format(test_loss_intensity/f))

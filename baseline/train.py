import torch
from dataset import SEGAN_Dataset
from tensorboardX import SummaryWriter
from hparams import hparams
from model import Generator, Discriminator
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
if __name__ == "__main__":
    
   
    
    # 定义device
    device = torch.device("cuda:0")
    
    # 导入参数
    para = hparams()
    
    # 创建数据保存文件夹
    os.makedirs(para.save_path,exist_ok=True)
    
    # 创建生成器
    generator = Generator()
    generator = generator.to(device)
    
    # 创建鉴别器
    discriminator = Discriminator()
    discriminator = discriminator.to(device)

    # 创建优化器
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)
    
    # 定义数据集
    m_dataset = SEGAN_Dataset(para)
    
    # 获取ref-batch
    ref_batch = m_dataset.ref_batch(para.ref_batch_size)
    ref_batch = Variable(ref_batch)
    ref_batch = ref_batch.to(device)
    
    writer=SummaryWriter("/data/lingtianze/zyy/denoise/segan/logs_train")
    # 定义dataloader
    m_dataloader = DataLoader(m_dataset,batch_size = para.batch_size,shuffle = True, num_workers = 4)
    loss_d_all =0
    loss_g_all =0
    n_step =0
    
    for epoch in range(para.n_epoch):
        clean_loss_total=0.
        noisy_loss_total=0.
        g_cond_loss_total=0.
        g_loss_total=0.
        for i_batch, sample_batch in enumerate(m_dataloader):
            batch_clean = sample_batch[0]
            batch_noisy = sample_batch[1]
            target=sample_batch[2]
            batch_clean = Variable(batch_clean)
            batch_noisy = Variable(batch_noisy)
            target= Variable(target)
            batch_clean = batch_clean.to(device)
            batch_noisy = batch_noisy.to(device)
            target = target.to(device)

            batch_z = nn.init.normal_(torch.Tensor(batch_clean.size(0), para.size_z[0], para.size_z[1]))
            batch_z = Variable(batch_z)
            batch_z = batch_z.to(device)
            
            
            discriminator.zero_grad()
            train_batch = Variable(torch.cat([batch_clean,batch_noisy],dim=1))
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            # clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(batch_noisy, batch_z)
            outputs = discriminator(torch.cat((generated_outputs, batch_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            # # noisy_loss.backward()

            d_loss = clean_loss + noisy_loss
            d_loss.backward()
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(batch_noisy, batch_z)
            gen_noise_pair = torch.cat((generated_outputs, batch_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(batch_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            
            g_loss = g_loss_ + g_cond_loss
            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()
            clean_loss_total+=clean_loss
            noisy_loss_total+=noisy_loss
            g_loss_total+=g_loss_
            g_cond_loss_total+=g_cond_loss
            
            print("Epoch %d:%d d_clean_loss %.4f, d_noisy_loss %.4f, g_loss %.4f, g_conditional_loss %.4f"%(epoch + 1,i_batch,clean_loss,noisy_loss,g_loss_,g_cond_loss))
        writer.add_scalar("d_clean_loss", clean_loss_total, epoch)
        writer.add_scalar("d_noisy_loss", noisy_loss_total, epoch)
        writer.add_scalar("g_loss", g_loss_total, epoch)
        writer.add_scalar("g_conditional_loss", g_cond_loss_total, epoch)
        
        g_model_name = os.path.join(para.path_save,"G_"+str(epoch)+"_%.4f"%(g_cond_loss)+".pkl")
        d_model_name = os.path.join(para.path_save,"D_"+str(epoch)+"_%.4f"%(noisy_loss)+".pkl")
        torch.save(generator.state_dict(), g_model_name)
        # torch.save(discriminator.state_dict(), d_model_name)

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
    
    
    
    
    
    
    
    
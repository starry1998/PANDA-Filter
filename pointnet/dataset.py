from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import math

def pad_to_length(data: list, length, pad_token=0.):
    for i in range(length - len(data)):
        data.append(pad_token)
class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 tolerance=25.0,
                 npoints=1500):
        self.npoints = npoints
        #root="/data/lingtianze/zyy/denoise/data/dataset"
        self.root = root
        self.tolerance=tolerance
        #分为几类
        self.num_seg_classes = 2
        #读取filepath
        self.catfile = os.path.join(self.root,'filepath.txt')
        self.cat = {}
        with open(self.catfile,'r') as f:
            for line in f:
                ls=line.strip().split()
                print('ls:',ls)
                fileindex_list=ls[0].split('_')
                fileindex=fileindex_list[1]+fileindex_list[3]
                self.cat[fileindex]=ls
        print("self.cat:{}".format(self.cat))
        print(len(self.cat))
        self.msms_noise_all=pd.DataFrame()
        self.msms_clean_all=pd.DataFrame()
        for k,v in self.cat.items():
            msms_noise=pd.read_pickle(v[0])
            # print(list(msms_noise))
            # print(msms_noise['scan_number'])
            msms_noise['scan_number']=msms_noise['scan_number'].str.strip()+'_'+k
            # print(msms_noise['scan_number'])
            # 
            
            # print(msms_noise.index)           
            msms_clean=pd.read_pickle(v[1])
            # print(list(msms_clean))
            # print(len(msms_clean))
            msms_clean['scan_number']=msms_clean['scan_number'].apply(str)
            msms_clean['scan_number']=msms_clean['scan_number']+'_'+k
            scannumber_list=msms_clean['scan_number'].tolist()
            msms_noise=msms_noise[msms_noise['scan_number'].isin(scannumber_list)]
            # print(len(msms_clean))
            msms_clean.set_index('scan_number',drop=False,inplace=True)
            msms_noise.set_index('scan_number',drop=False,inplace=True)
            # print(msms_clean.index)
            self.msms_noise_all=self.msms_noise_all.append(msms_noise)
            # print(self.msms_noise_all.index)
            self.msms_clean_all=self.msms_clean_all.append(msms_clean)
            # print(self.msms_clean_all.index)
        self.msms_index=self.msms_clean_all.index.tolist()
        # print(len(self.msms_index))
        
    def __getitem__(self, index):
        scan_number=self.msms_index[index]
        spectrum_noise=self.msms_noise_all.loc[scan_number,:]
        spectrum_clean=self.msms_clean_all.loc[scan_number,:]     
        mz_label=0
        mz_label_list=[]
        intensity_label_list=[]
        mz_noise_list=spectrum_noise['mz']
        # print("mz_noise:{}".format(mz_noise_list))
        # print(len(mz_noise_list))
        mz_clean_str=spectrum_clean['mz']
        mz_clean_list=mz_clean_str.split(';')
        # print("mz_clean_list:{}".format(mz_clean_list))
        # print(len(mz_clean_list))
        intensity_noise_list=spectrum_noise['intensities']
        # print("intensity_noise_list:{}".format(intensity_noise_list))
        # print(len(intensity_noise_list))
        intensity_clean_str=spectrum_clean['spectrum_pred']
        intensity_clean_list=intensity_clean_str.split(';')
        # print("intensity_clean_list:{}".format(intensity_clean_list))
        # print(len(intensity_clean_list))
        # sequence=spectrum_clean['pr_id']
        # print(sequence)
        #遍历对应的m/z预测结果；如果找到标签为1，没有找到标签为0
        #遍历对应的intensity预测结果：如果标签为1，保存对应的intensity的值，如果标签为0，intensity=0
        label1_number=0
        for mz_noise in mz_noise_list:
            location=0
            # print(type(mz_noise))
            # print(len(mz_label_list))
            for mz_clean in mz_clean_list:
                mz_clean=float(mz_clean)
                # print(type(mz_clean))
                #fabs(noise_mz-clean_mz)/clean_mz*1e6
                if math.fabs(mz_noise-mz_clean)/mz_clean*1e6<=self.tolerance:
                    # print("Find the peak!:{}".format(location))
                    label1_number=label1_number+1
                    mz_label=1
                    mz_label_list.append(mz_label)
                    intensity_label=intensity_clean_list[location]
                    # print(intensity_label)
                    intensity_label_list.append(float(intensity_label))
                    break
                else:
                    mz_label=0
                    location=location+1
            if(mz_label==0):
                mz_label_list.append(mz_label)
                intensity_label_list.append(mz_label)
        # print("mz_label_list:{}".format(mz_label_list))
        # print(len(mz_label_list))
        label_rate=label1_number/len(mz_clean_list)
        # print("intenisty_label_list:{}".format(intensity_label_list))
        # print(len(intensity_label_list))
        #padding
        pad_to_length(mz_noise_list,self.npoints)
        pad_to_length(intensity_noise_list,self.npoints)
        pad_to_length(mz_label_list,self.npoints)
        pad_to_length(intensity_label_list,self.npoints)
        mz_set=np.array(mz_noise_list,dtype=np.float32)
        intensity_set=np.array(intensity_noise_list,dtype=np.float32)
        mz_label_set=np.array(mz_label_list,dtype=np.int64)
        intensity_label_set=np.array(intensity_label_list,dtype=np.float32)
        #标准化强度值
        intensity_max=np.max(intensity_set)
        norm_intensity_set=intensity_set/intensity_max
        # print("norm_intensity:{}",norm_intensity_set)
        mz_set=np.expand_dims(mz_set,axis=1)
        norm_intensity_set=np.expand_dims(norm_intensity_set,axis=1)
        mz_label_set=np.expand_dims(mz_label_set,axis=1)
        intensity_label_set=np.expand_dims(intensity_label_set,axis=1)
        point_set=np.concatenate((mz_set,norm_intensity_set),axis=1)
        # seg=np.concatenate((mz_label_set,intensity_label_set),axis=1)
        # print(point_set.shape)
        # print("point_set:{}____seg:{}".format(point_set,seg))
        point_set = torch.from_numpy(point_set)
        mz_label_set = torch.from_numpy(mz_label_set)
        intensity_label_set=torch.from_numpy(intensity_label_set)
        return point_set, mz_label_set, intensity_label_set

    def __len__(self):
        print('len(self.msms_index):{}'.format(len(self.msms_index)))
        return len(self.msms_index)
    
    
if __name__ == '__main__':
    
    datapath = '/data/lingtianze/zyy/denoise/data/dataset'
    d = ShapeNetDataset(root = datapath)
    print(len(d))
    # label_r=0
    # for i in range(len(d)):
    #     ps, mz_label, intensity_label, label_rate = d[i]
    #     # print("ps:{}".format(ps))
    #     # print("seg:{}".format(seg))
    #     label_r=label_r+label_rate
    # print(label_r/len(d))
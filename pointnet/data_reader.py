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

def pad_to_length(data, length, pad_token=0.):
    data=data.tolist()
    for i in range(length - len(data)):
        data.append(pad_token)
    data=np.array(data)
    return data



class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 datapath,
                 tolerance=25.0,
                 npoints=500,
                 ):
        self.npoints = npoints
        #root="/data/lingtianze/zyy/denoise/data/dataset"
        self.datapath = datapath
        self.tolerance=tolerance
        #分为几类
        self.num_seg_classes = 2
        # 读取多个数据集文件
        # msms_file_path1=os.path.join(datapath,"traindata_3xHCD_25_sn01.pkl")
        # msms_file_path2=os.path.join(datapath,"traindata_3xHCD_30_sn01.pkl")
        # msms_file_path3=os.path.join(datapath,"traindata_3xHCD_35_sn01.pkl")
        
        # msms_file1=pd.read_pickle(msms_file_path1)
        # msms_file2=pd.read_pickle(msms_file_path2)
        # msms_file3=pd.read_pickle(msms_file_path3)
        # self.msms_file=pd.concat([msms_file1,msms_file2,msms_file3])
        # #读取单个文件
        # msms_file_path=os.path.join(datapath,"PXD010595_25.pkl")
        self.msms_file=pd.read_pickle(datapath)
        
        #这个数据集的标记率
        # label_rate_mean=self.msms_file['label_rate'].mean()
        # print(label_rate_mean)
        # rn_ratio=self.msms_file['sn_rate'].mean()
        # print(rn_ratio)
        #根据label_rate和peak_number进行数据筛选
        # self.msms_file=self.msms_file[self.msms_file["label_rate"]>0.8]
        # self.msms_file=self.msms_file[self.msms_file["sn_rate"]>0.1]
        # print(self.msms_file.shape) 
        # self.msms_file=self.msms_file[self.msms_file["peak_number"]>200]
        # print(self.msms_file.shape)
        #保存scan_number_list
        self.msms_file.set_index('scan_number_noise', drop=False, inplace=True)
        self.msms_index=self.msms_file.index.tolist()
        
    def __getitem__(self, index):
        scan_number=self.msms_index[index]
        spectrum=self.msms_file.loc[scan_number,:]
        
        #获取实验谱图的point信息和标记信息
        mz_noise_list=spectrum['mz_noise']
        intensity_noise_list=spectrum['intensities_noise']
        mz_label_list=spectrum['mz_label']
        intensity_label_list=spectrum['intensity_label']
        #获取理论谱图的point信息
        mz_clean_list=spectrum['mz_clean'].split(";")
        intensity_clean_list=spectrum['intensities_clean'].split(";")
        
        #获取每个点的特征
        comp_list=spectrum['comp_list']
        sister_list=spectrum['sister_list']
        #list转为numpy
        mz_noise_set=np.array(mz_noise_list, dtype=np.float32)
        intensity_noise_set=np.array(intensity_noise_list, dtype=np.float32)
        mz_label_set=np.array(mz_label_list,dtype=np.int64)
        intensity_label_set=np.array(intensity_label_list, dtype=np.float32)
        mz_clean_set=np.array(mz_clean_list, dtype=np.float32)
        intensity_clean_set=np.array(intensity_clean_list, dtype=np.float32 )
        comp_set=np.array(comp_list,dtype=np.float32)
        sister_set=np.array(sister_list, dtype=np.float32)
        # print(np.count_nonzero(mz_label_set))
        #得到强度top500的peak
        if(len(intensity_noise_set)>500):
            intensity_sorted=np.argsort(-intensity_noise_set)
            intensity_sorted=np.delete(intensity_sorted, np.s_[500:])
            intensity_noise_set=intensity_noise_set[intensity_sorted] 
            mz_noise_set=mz_noise_set[intensity_sorted]
            mz_label_set=mz_label_set[intensity_sorted]
            intensity_label_set=intensity_label_set[intensity_sorted]
            comp_set=comp_set[intensity_sorted]
            sister_set=sister_set[intensity_sorted]
            # print(np.count_nonzero(mz_label_set))
        else:    
        #对数据进行padding
            mz_noise_set=pad_to_length(mz_noise_set,self.npoints)
            intensity_noise_set=pad_to_length(intensity_noise_set, self.npoints)
            mz_label_set=pad_to_length(mz_label_set, self.npoints, pad_token=-100)
            intensity_label_set=pad_to_length(intensity_label_set, self.npoints, pad_token=-100)
            comp_set=pad_to_length(comp_set, self.npoints)
            sister_set=pad_to_length(sister_set,self.npoints,[0,0,0,0,0,0,0,0,0,0])
        mz_clean_set=pad_to_length(mz_clean_set, self.npoints)
        intensity_clean_set=pad_to_length(intensity_clean_set, self.npoints)
        #标准化强度值
        intensity_noise_max=np.max(intensity_noise_set)
        norm_intensity_noise_set=intensity_noise_set/intensity_noise_max
        mz_noise_set=np.expand_dims(mz_noise_set,axis=1)
        norm_intensity_noise_set=np.expand_dims(norm_intensity_noise_set,axis=1)
        comp_set=np.expand_dims(comp_set,axis=1)
        mz_label_set=np.expand_dims(mz_label_set,axis=1)
        intensity_label_set=np.expand_dims(intensity_label_set,axis=1)
        mz_clean_set=np.expand_dims(mz_clean_set,axis=1)
        intensity_clean_set=np.expand_dims(intensity_clean_set,axis=1)
        # point_noise_set=np.concatenate((mz_noise_set, norm_intensity_noise_set),axis=1)
        point_noise_feature_set=np.concatenate((mz_noise_set, norm_intensity_noise_set, comp_set, sister_set),axis=1)
        point_clean_set=np.concatenate((mz_clean_set,intensity_clean_set),axis=1)
        # point_noise_set=point_noise_set.astype(np.float32)
        point_noise_feature_set=point_noise_feature_set.astype(np.float32)
        point_clean_set=point_clean_set.astype(np.float32)
        mz_label_set=mz_label_set.astype(np.float32)
        intensity_label_set=intensity_label_set.astype(np.float32)
        
        point_noise_feature_set = torch.from_numpy(point_noise_feature_set)
        point_clean_set = torch.from_numpy(point_clean_set)
        mz_label_set = torch.from_numpy(mz_label_set)
        intensity_label_set=torch.from_numpy(intensity_label_set)
        return point_noise_feature_set, point_clean_set, mz_label_set, intensity_label_set
    
    def __len__(self):
        # print('len(self.msms_index):{}'.format(len(self.msms_index)))
        return len(self.msms_index)

if __name__ == '__main__':
    datapath = '/data/lingtianze/zyy/denoise/data/dataset/traindata_set/traindata_nce25_sn02.pkl'
    d = ShapeNetDataset(datapath)
    dataset_size=len(d)
    print(dataset_size)
    print("train_dataset:{}".format(0.8*dataset_size))
    print("val_dataset:{}".format(0.1*dataset_size))
    print("test_dataset:{}".format(dataset_size-0.1*dataset_size-0.8*dataset_size))
    # Returns False because the first key is false.
    # For dictionaries the all() function checks the keys, not the values.
    
    # for i in range(len(d)):
    #     point_noise_feature_set , point_clean_set, mz_label_set, intensity_label_set=d[i]
    #     print(point_noise_feature_set.shape)
    #     print(point_clean_set.shape)
# if (self.discriminator == True):
        #     point_noise_set=np.expand_dims(point_noise_set,axis=0)
        #     point_clean_set=np.expand_dims(point_clean_set,axis=0)
        #     point_set=np.concatenate((point_noise_set,point_clean_set), axis=0)
        #     target=np.array([0,1])
        #     point_set=torch.from_numpy(point_set)
        #     target=torch.from_numpy(target)
        #     return point_set,target
        # else:
        # point_noise_set=torch.from_numpy(point_noise_set)
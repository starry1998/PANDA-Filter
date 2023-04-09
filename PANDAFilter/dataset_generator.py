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
from dataclasses import dataclass
import itertools


class TrainDataset():
    def __init__(self,
                 root,
                 tolerance=25.0,
                 ):
        self.root = root
        self.tolerance=tolerance
        #读取filepath并存储为字典
        self.catfile = self.root
        self.cat = {}
        with open(self.catfile,'r') as f:
            for line in f:
                ls=line.strip().split()
                fileindex_list=ls[0].split('_')
                methodname=fileindex_list[7].split('.')
                fileindex=methodname[0]+fileindex_list[2]+fileindex_list[4]
                # fileindex_list=ls[0].split('-')
                # methodname=fileindex_list[1].split('_')
                # fileindex=fileindex_list[2]+methodname[2]+methodname[3]
                self.cat[fileindex]=ls
        print("self.cat:{}".format(self.cat))
        print(len(self.cat))
        #获取所有质谱数据中的谱图
        self.msms_noise_all=pd.DataFrame()
        self.msms_clean_all=pd.DataFrame()
        for k,v in self.cat.items():
            msms_noise=pd.read_pickle(v[0])
            msms_noise['scan_number']=msms_noise['scan_number'].str.strip()+'_'+k
            msms_clean=pd.read_pickle(v[1])
            msms_clean['scan_number']=msms_clean['scan_number'].apply(str)
            msms_clean['scan_number']=msms_clean['scan_number']+'_'+k
            msms_clean.rename(columns={'spectrum_pred':'intensities'},inplace=True)
            
            #获取理论谱图的scannumber
            scannumber_list=msms_clean['scan_number'].tolist()
            #获取和理论谱图对应的实验谱图
            msms_noise=msms_noise[msms_noise['scan_number'].isin(scannumber_list)]
            #获取和实验谱图对应的理论谱图
            msms_noise_sn=msms_noise['scan_number'].tolist()
            msms_clean=msms_clean[msms_clean['scan_number'].isin(msms_noise_sn)]
           
            #将scan_number设置成数据集的索引
            msms_noise.set_index('scan_number',drop=False,inplace=True)
            msms_clean.set_index('scan_number',drop=False,inplace=True)    
            self.msms_noise_all=self.msms_noise_all.append(msms_noise)
            self.msms_clean_all=self.msms_clean_all.append(msms_clean)
        self.msms_index=self.msms_clean_all.index.tolist()
        print(len(self.msms_clean_all))
    #给谱图加标签
    def add_label(self,mz_noise_list,mz_clean_list,intensity_clean_list):
        mz_label_list=[]
        intensity_label_list=[]
        label1_number=0
        #遍历实验谱图的所有peak
        for mz_noise in mz_noise_list:
            location=0
            #遍历理论谱图的所有peak
            for mz_clean in mz_clean_list:
                mz_clean=float(mz_clean)
                #判断理论谱图的peak和实验谱图的peak是否在质量容差范围内，如果是，标记1，并退出标记；如果不是，标记0
                if math.fabs(mz_noise-mz_clean)/mz_clean*1e6<=self.tolerance:
                    label1_number=label1_number+1
                    mz_label=1
                    mz_label_list.append(mz_label)
                    intensity_label=intensity_clean_list[location]
                    intensity_label_list.append(float(intensity_label))
                    break
                else:
                    mz_label=0
                    location=location+1
            if(mz_label==0):
                mz_label_list.append(mz_label)
                intensity_label_list.append(float(mz_label))
        #计算标记率=标记为1的peak number除以理论谱所有的peak number
        label_rate=label1_number/len(mz_clean_list)
        signal_noise_rate=label1_number/len(mz_noise_list)
        return mz_label_list, intensity_label_list, label_rate, signal_noise_rate
    
    def calculate_features(self,mz,pre_mass,sister=np.array([-2, -1 ,1, 2, 17, 18, 28, 34, 35, 36])):
        mz_pair=list(itertools.permutations(mz, 2))
        mz_pair = np.array(mz_pair,dtype=np.float32)
        mz_pair_d = [mz_pair[i][0]- mz_pair[i][1] for i in range(len(mz_pair))]
        mz_sister = np.array([abs(d-sister) for d in mz_pair_d])#计算姐妹离子
        mz_comp = np.array([abs(sum(mz_pair[i])-pre_mass) for i in range(len(mz_pair))])#计算互补离子
        
        sister_location=np.array(np.where(mz_sister<0.5))
        comp_location=np.array(np.where(mz_comp<2.5))
        
        #创建一个sister_ion保存每个峰的sister ion
        array_shape=np.arange(len(sister)* len(mz))
        array_shape=array_shape.reshape(-1,len(sister))
        sister_ion=np.zeros_like(array_shape)
        #创建一个comp_ion保存每个峰的comp ion
        comp_ion=np.zeros_like(mz)
        for i in range(len(sister_location[0])):
            peakindex_sister=sister_location[0][i]//(len(mz)-1)
            sister_ion[peakindex_sister][sister_location[1][i]]=1
        for j in range(len(comp_location[0])):
            peakindex_comp=comp_location[0][j]//(len(mz)-1)
            comp_ion[peakindex_comp]=1
        return sister_ion, comp_ion

    def process_spectrum(self): 
        scan_number_list=[]
        peak_number_list=[]
        mz_label=[]
        intensity_label=[]
        label_rate_list=[]
        sn_ratio_list=[]
        comp_list=[]
        sister_list=[]
        for i,scan_number in enumerate(self.msms_index):
            spectrum_noise=self.msms_noise_all.loc[scan_number,:]
            spectrum_clean=self.msms_clean_all.loc[scan_number,:]
            mz_noise_list=spectrum_noise['mz']
            peak_number=len(mz_noise_list)
            mz_clean_str=spectrum_clean['mz']
            mz_clean_list=mz_clean_str.split(';')
            # intensity_noise_list=spectrum_noise['intensities']
            intensity_clean_str=spectrum_clean['intensities']
            intensity_clean_list=intensity_clean_str.split(';')
            #肽段的质量
            pep_mass=np.array(spectrum_noise['mass'],dtype=np.float32)*int(spectrum_clean['pr_id'][-1])
            #给谱图中的每个peak加标签
            mz_label_list, intensity_label_list, label_rate, signal_noise_ratio=self.add_label(mz_noise_list,mz_clean_list,intensity_clean_list)
            
            
            if(label_rate>0.3 and label_rate<=0.8):
                scan_number_list.append(scan_number)
                peak_number_list.append(peak_number)
                mz_label.append(mz_label_list)
                intensity_label.append(intensity_label_list)
                label_rate_list.append(label_rate)
                sn_ratio_list.append(signal_noise_ratio)
                print("——————计算离子特征————————")
                sister_ion, comp_ion=self.calculate_features(np.array(mz_noise_list),pep_mass) #计算谱图中的每个peak是否有互补离子
                sister_list.append(sister_ion)
                comp_list.append(comp_ion)
            # scan_number_list.append(scan_number)
            # peak_number_list.append(peak_number)
            # mz_label.append(mz_label_list)
            # intensity_label.append(intensity_label_list)
            # label_rate_list.append(label_rate)
            # sn_ratio_list.append(signal_noise_ratio)
            # print("——————计算离子特征————————")
            # sister_ion, comp_ion=self.calculate_features(np.array(mz_noise_list),pep_mass) #计算谱图中的每个peak是否有互补离子
            # sister_list.append(sister_ion)
            # comp_list.append(comp_ion)
        print("——————所有谱图处理完毕————————")
        df_label=pd.DataFrame({'scan_number':scan_number_list,
                               'peak_number':peak_number_list,
                               'mz_label':mz_label,
                               'intensity_label':intensity_label,
                               'label_rate':label_rate_list,
                               'sn_rate':sn_ratio_list,
                               'sister_list':sister_list,
                               'comp_list':comp_list
                               })
        df_label.set_index('scan_number',drop=True,inplace=True)
        df_msms=self.msms_noise_all.join(df_label, how='inner')
        df=df_msms.join(self.msms_clean_all, lsuffix='_noise', rsuffix='_clean',how='inner')
        return df
    def __len__(self):
        print('len(TrainData):{}'.format(len(self.msms_index)))
        return len(self.msms_index)
    
    
if __name__ == '__main__':
    datapath = '/data/lingtianze/zyy/denoise/data/dataset/PXD004732_nce35.txt'
    d = TrainDataset(root = datapath)
    data=d.process_spectrum()
    data.to_pickle("/data/lingtianze/zyy/denoise/data/dataset/traindata_set/PXD004732_35_0308.pkl")
    
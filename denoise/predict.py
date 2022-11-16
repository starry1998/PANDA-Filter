from model import *
import torch
import pandas as pd
from data_reader import ShapeNetDataset
import os
import itertools
import argparse
from pathlib import Path
#数据处理
#模型预测
def predict(tensor):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Generator=PointNetDenseCls(d=13, k=2, feature_transform=False)
    Generator.load_state_dict(torch.load("/data/lingtianze/zyy/denoise/seg_model/trained/nce25_sn01/generator_model_450.pth",map_location='cpu'))
    
    Generator = Generator.to(device)
    Generator.eval()
    
    #获取数据
    # dataset=ShapeNetDataset("/data/lingtianze/zyy/denoise/data/dataset/traindata_set/traindata_3xHCD_30_feature.pkl")
    # points_noise, _ , mz_target , intensity_target =dataset[0]
    tensor = tensor.unsqueeze(0)
    tensor = tensor.transpose(2,1)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        mz_pred, intensity_pred, _, _ = Generator(tensor)
        mz_pred = mz_pred.view(-1, 2)
        mz_pred_choice = mz_pred.data.max(1)[1]
        mz_mask=mz_pred_choice.unsqueeze(0)
        mz_mask=mz_mask.view(-1,500)
        intensity_mask=mz_mask.unsqueeze(2)
        spectrum_noise_mz=tensor[:, 0, :]
        #预测得到的m/z
        spectrum_noise_mz=spectrum_noise_mz.masked_fill(mz_mask==0, 0)
        #预测得到的intensity
        intensity_pred=intensity_pred.masked_fill(intensity_mask==0, 0)
        #得到预测的谱图
        spectrum_noise_mz=spectrum_noise_mz.view(500)
        intensity_pred=intensity_pred.view(500)
        return spectrum_noise_mz, intensity_pred
    
def calculate_features(mz,pre_mass,sister=np.array([-2, -1 ,1, 2, 17, 18, 28, 34, 35, 36])):
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
    
def pad_to_length(data, length, pad_token=0.):
    data=data.tolist()
    for i in range(length - len(data)):
        data.append(pad_token)
    data=np.array(data)
    return data

def tensorize(mz_noise_list,intensity_noise_list,mass):
    mz_noise_set=np.array(mz_noise_list, dtype=np.float32)
    intensity_noise_set=np.array(intensity_noise_list, dtype=np.float32)
    sister_list, comp_list=calculate_features(mz_noise_set, mass)
    comp_set=np.array(comp_list,dtype=np.float32)
    sister_set=np.array(sister_list, dtype=np.float32)  
    if(len(intensity_noise_set)>500):
        intensity_sorted=np.argsort(-intensity_noise_set)
        intensity_sorted=np.delete(intensity_sorted, np.s_[500:])
        intensity_noise_set=intensity_noise_set[intensity_sorted] 
        mz_noise_set=mz_noise_set[intensity_sorted]
        comp_set=comp_set[intensity_sorted]
        sister_set=sister_set[intensity_sorted]
    else:
        mz_noise_set=pad_to_length(mz_noise_set,500)#对数据进行padding
        intensity_noise_set=pad_to_length(intensity_noise_set, 500)
        comp_set=pad_to_length(comp_set, 500)
        sister_set=pad_to_length(sister_set, 500,[0,0,0,0,0,0,0,0,0,0])
    #标准化强度值
    intensity_noise_max=np.max(intensity_noise_set)
    norm_intensity_noise_set=intensity_noise_set/intensity_noise_max
    mz_noise_set=np.expand_dims(mz_noise_set,axis=1)
    norm_intensity_noise_set=np.expand_dims(norm_intensity_noise_set,axis=1)    
    comp_set=np.expand_dims(comp_set,axis=1)
    point_noise_feature_set=np.concatenate((mz_noise_set, norm_intensity_noise_set, comp_set, sister_set),axis=1)
    point_noise_feature_set = torch.from_numpy(point_noise_feature_set)
    point_noise_feature_set=point_noise_feature_set.type(torch.FloatTensor)
    return point_noise_feature_set

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--msmspath', type=str, default="/data/lingtianze/zyy/denoise/workplace/spectrum_noise/01974c_BG1-TUM_missing_first_7_01_01-3xHCD-1h-R4.pkl", help="msms path")
    # msms_path="/data/lingtianze/zyy/denoise/workplace/spectrum_noise/01974c_BG1-TUM_missing_first_7_01_01-3xHCD-1h-R4.pkl"
    opt = parser.parse_args()
    msmspath=Path(opt.msmspath)
    dataset=pd.read_pickle(msmspath)
    out_dir= os.path.join("/data/lingtianze/zyy/denoise/workplace",'peakFilter',(msmspath.stem + '.pkl'))
    dataset.set_index('scan_number',drop=False,inplace=True)
    spectr_index=dataset.index.tolist()
    mzs_list=[]
    intensities_list=[]
    for s in spectr_index:
        spectrum_noise=dataset.loc[s,:]
        mz_list=spectrum_noise['mz']
        intensity_list=spectrum_noise['intensities']
        m_over_z=np.array(spectrum_noise['mass'],dtype=np.float32)
        charge=int(spectrum_noise['charge'][0])
        mass=m_over_z*charge
        point_noise_feature_set=tensorize(mz_list,intensity_list,mass)
        spectrum_mz, spectrum_intensity=predict(point_noise_feature_set)
        spectrum_mz=spectrum_mz.data.cpu().numpy()
        spectrum_intensity=spectrum_intensity.data.cpu().numpy()
        mzs_list.append(spectrum_mz)
        intensities_list.append(spectrum_intensity)
        
    #输出结果的保存
    df_result=pd.DataFrame()
    df_result['title']=dataset['title']
    df_result['scan_number']=dataset['scan_number']
    df_result['rtinseconds']=dataset['rtinseconds']
    df_result['pepmass']=dataset['pepmass']
    df_result['charge']=dataset['charge']
    df_result['mz']=mzs_list
    df_result['intensity']=intensities_list
    df_result.to_pickle(out_dir)
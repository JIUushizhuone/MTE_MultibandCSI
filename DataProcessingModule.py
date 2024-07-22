import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import torch
import math
import torch.nn.functional as F
from torch import nn
# Load data
def LoadData(scenario_name, EM_index_ori, EM_index_compare, device):
    data_all_ori = []
    data_all_compare = []
    for rx_id in range(1, 5): # rx_N = 4
        # Load the data for the current rx_id
        data_path_ori = f'data0329/{scenario_name}/{scenario_name}{EM_index_ori}/rx{rx_id}/{scenario_name}{EM_index_ori}_rx{rx_id}_CTF.pt'
        data_path_compare = f'data0329/{scenario_name}/{scenario_name}{EM_index_compare}/rx{rx_id}/{scenario_name}{EM_index_compare}_rx{rx_id}_CTF.pt'
        
        data_ori = torch.load(data_path_ori)
        data_compare = torch.load(data_path_compare)

        # Append the current data to the overall data
        data_all_ori.append(data_ori.unsqueeze(0))  # Add an extra dimension for rx_id
        data_all_compare.append(data_compare.unsqueeze(0))  # Add an extra dimension for rx_id

    # Convert the list to a torch.Tensor
    data_all_ori = torch.cat(data_all_ori, dim=0)
    data_all_compare = torch.cat(data_all_compare, dim=0)

    return data_all_ori, data_all_compare


# One-hot encoding for the labels
def convert_to_one_hot(Y_data):
    # Convert the 3D coordinates to string
    coords_str = np.array([''.join(map(str, coord)) for coord in Y_data])

    # Create a mapping from coordinate to unique label
    coord_to_label = {coord: i for i, coord in enumerate(np.unique(coords_str))}

    # Convert the coordinates to labels
    labels = np.array([coord_to_label[coord] for coord in coords_str])

    # Create one-hot encoding
    one_hot_encoded = np.eye(len(coord_to_label))[labels]

    return one_hot_encoded

class CustomLoss_L1(nn.Module):
    def __init__(self):
        super(CustomLoss_L1, self).__init__()

    def forward(self, outputs, targets, s_name,device):
        loss_mat = outputs - targets
        if s_name == 'sy':
            weight = torch.tensor([1,0.75,1])
        if s_name == 'sy_all':
            weight = torch.tensor([0.25,0.25,0.5])
        if s_name == 'xh':
            weight = torch.tensor([1.1,1,1])
        if s_name == 'xh_all':
            weight = torch.tensor([0.275, 0.25, 0.25])
        loss_mat[:,0] = loss_mat[:,0] * weight[0]
        loss_mat[:,1] = loss_mat[:,1] * weight[1]
        loss_mat[:,2] = loss_mat[:,2] * weight[2]
        loss_mat = loss_mat.abs() 
        loss = loss_mat.mean()# L1 norm
        # loss = torch.sqrt(torch.mean(loss_mat**2)) # L2 norm
        var_x = loss_mat[:,0].var(); var_y = loss_mat[:,1].var(); var_z = loss_mat[:,2].var()
        mean_x = loss_mat[:,0].mean(); mean_y = loss_mat[:,1].mean(); mean_z = loss_mat[:,2].mean()
        zero_row = torch.zeros(loss_mat.size(1)); zero_row = zero_row.to(device)
        zero_rows = (loss_mat.eq(zero_row)).all(dim=1) 
        num_rate = zero_rows.sum()/loss_mat.shape[0] # Number of zero rows (correct localization)
        return loss, var_x, var_y, var_z, num_rate, mean_x, mean_y, mean_z
    
class CustomLoss_L2(nn.Module):
    def __init__(self):
        super(CustomLoss_L2, self).__init__()

    def forward(self, outputs, targets, s_name, device):
        loss_mat = outputs - targets
        if s_name == 'sy':
            weight = torch.tensor([1,0.75,1])
        if s_name == 'sy_all':
            weight = torch.tensor([0.25,0.25,0.5])
        if s_name == 'xh':
            weight = torch.tensor([1.1,1,1])
        if s_name == 'xh_all':
            weight = torch.tensor([0.275, 0.25, 0.25])
        loss_mat[:,0] = loss_mat[:,0] * weight[0]
        loss_mat[:,1] = loss_mat[:,1] * weight[1]
        loss_mat[:,2] = loss_mat[:,2] * weight[2]
        loss_mat = loss_mat.abs() 
        
        # loss = loss_mat.mean()# L1 norm
        loss = torch.sqrt(torch.mean(loss_mat**2)) # L2 norm
        var_x = loss_mat[:,0].var(); var_y = loss_mat[:,1].var(); var_z = loss_mat[:,2].var()
        mean_x = loss_mat[:,0].mean(); mean_y = loss_mat[:,1].mean(); mean_z = loss_mat[:,2].mean()
        zero_row = torch.zeros(loss_mat.size(1)); zero_row = zero_row.to(device)
        zero_rows = (loss_mat.eq(zero_row)).all(dim=1) 
        num_rate = zero_rows.sum()/loss_mat.shape[0] # Number of zero rows (correct localization)
        return loss, var_x, var_y, var_z, num_rate, mean_x, mean_y, mean_z


def CDF_save(outputs, targets, s_name, log_dir):
        loss_mat = outputs - targets
        if s_name == 'sy':
            weight = torch.tensor([1,0.75,1])
        if s_name == 'sy_all':
            weight = torch.tensor([0.25,0.25,0.5])
        if s_name == 'xh':
            weight = torch.tensor([1.1,1,1])
        if s_name == 'xh_all':
            weight = torch.tensor([0.275, 0.25, 0.25])
        loss_mat[:,0] = loss_mat[:,0] * weight[0]
        loss_mat[:,1] = loss_mat[:,1] * weight[1]
        loss_mat[:,2] = loss_mat[:,2] * weight[2]
        loss_mat = loss_mat.abs() 
        
        rmse_each_row = torch.sqrt(torch.mean(loss_mat**2, dim=1)).cpu().detach().numpy()

        # Save RMSE to a CSV file
        df_rmse = pd.DataFrame(rmse_each_row, columns=['RMSE'])
        df_rmse.to_csv(log_dir+'.csv', index=False)

def scale_down(data,x_range,y_range,z_range):
    data[:,0] = data[:,0]/x_range
    data[:,1] = data[:,1]/y_range
    data[:,2] = data[:,2]/z_range
    return data

def scale_up(data,x_range,y_range,z_range):
    data[:,0] = data[:,0]*x_range
    data[:,1] = data[:,1]*y_range
    data[:,2] = data[:,2]*z_range
    return data

# Gaussian white noise, P = 8.0975e-06
def GaussianNoise(CTF_complex,sigma, device = 'cuda:1'):
    # print(CTF_complex)
    CTF_complex_N = torch.randn(CTF_complex.shape)*sigma + torch.randn(CTF_complex.shape)*sigma*1j + CTF_complex
    # print(CTF_complex_N)
    return CTF_complex_N

def GaussianNoise_abs(CTF,sigma, device = 'cuda:1'):
    # print(CTF_complex)
    CTF = torch.randn(CTF.shape)*sigma + CTF
    # print(CTF_complex_N)
    return CTF

def PhaseNoise(CTF_complex, device = 'cuda:1'):# Useless
    CTF_complex = torch.mul(CTF_complex,torch.exp(torch.randn(CTF_complex.shape)*1j))
    return CTF_complex

def DeviceHeterogenity(CTF_complex, HeteroRatio,  device = 'cuda:1'):
    CTF_complex_DH = torch.mul((1+torch.randn(CTF_complex.shape)*HeteroRatio),CTF_complex)
    return CTF_complex_DH


# Noise_add (dB) 
def Noise_add(CTF, sigma, HeteroRatio): 
    CTF_Noise = CTF
    CTF_Noise = GaussianNoise_abs(CTF_Noise, sigma)
    CTF_Noise = DeviceHeterogenity(CTF_Noise, HeteroRatio)
    
    return CTF_Noise

def Add_Gaussian_Noise(data, sigma_dB, HeteroRatio):
    mean_value = data.mean(dim = 1)
    for i in np.arange(data.shape[1]):
        Noise_Gaussian = mean_value[i]*sigma_dB; 
        data[:,i,:] = Noise_add(data[:,i,:],Noise_Gaussian, HeteroRatio) 
    return data

def Standardization(data_to_process):
    for i in np.arange(int(data_to_process.shape[2])/256):
        i = int(i)
        mu = torch.tensor(data_to_process[:,:,i*256:(i+1)*256].mean(dim=2, keepdim=True))
        std = torch.tensor(data_to_process[:,:,i*256:(i+1)*256].std(dim=2, keepdim=True))
        data_to_process[:,:,i*256:(i+1)*256]= (data_to_process[:,:,i*256:(i+1)*256] - mu) / std
    return data_to_process

# def Normalization(data_to_process):
#     for i in range(data_to_process.shape[0]):
#         for j in range(data_to_process.shape[1]):
#             for k in np.arange(int(data_to_process.shape[2])/256):
#                 k = int(k)
#                 min_val = torch.min(data_to_process[i,j,k*256:(k+1)*256])
#                 max_val = torch.max(data_to_process[i,j,k*256:(k+1)*256])
#                 data_to_process[i,j,k*256:(k+1)*256] = 2 * (data_to_process[i,j,k*256:(k+1)*256] - min_val) / (max_val - min_val) - 1
#     return data_to_process


def Normalization(data_to_process):
    # 将数据重塑为(-1, 256)的形状，每256个元素作为一个子集
    data_reshaped = data_to_process.reshape(-1, 256)

    # 计算每个子集的最小值和最大值
    min_val = torch.min(data_reshaped, dim=1, keepdim=True)[0]
    max_val = torch.max(data_reshaped, dim=1, keepdim=True)[0]

    # 对每个子集进行归一化处理
    data_normalized = 2 * (data_reshaped - min_val) / (max_val - min_val) - 1

    # 将数据恢复到原始的形状
    data_to_process = data_normalized.reshape(data_to_process.shape)

    return data_to_process

def sliding_average(data, window_size):
    N, C, L = data.shape
    assert L >= window_size, "The length of the data must be greater than or equal to the window size."
    unfolded_data = data.unfold(dimension=2, size=window_size, step=1)
    sliding_avg = unfolded_data.mean(dim=-1)
    pad_left = sliding_avg[:, :, 0].unsqueeze(-1).expand(-1, -1, window_size // 2)
    pad_right = sliding_avg[:, :, -1].unsqueeze(-1).expand(-1, -1, window_size // 2)
    sliding_avg = torch.cat((pad_left, sliding_avg, pad_right), dim=-1)
    if window_size % 2 == 1:
        sliding_avg = torch.cat((sliding_avg, sliding_avg[:, :, -1].unsqueeze(-1)), dim=-1)
    # Only select the first 2048 elements
    sliding_avg = sliding_avg[:, :, :L]
    return sliding_avg
def sliding_average_1024(data, window_size):
    N, C, L = data.shape
    assert L >= window_size, "The length of the data must be greater than or equal to the window size."
    unfolded_data = data.unfold(dimension=2, size=window_size, step=1)
    sliding_avg = unfolded_data.mean(dim=-1)
    pad_left = sliding_avg[:, :, 0].unsqueeze(-1).expand(-1, -1, window_size // 2)
    pad_right = sliding_avg[:, :, -1].unsqueeze(-1).expand(-1, -1, window_size // 2)
    sliding_avg = torch.cat((pad_left, sliding_avg, pad_right), dim=-1)
    if window_size % 2 == 1:
        sliding_avg = torch.cat((sliding_avg, sliding_avg[:, :, -1].unsqueeze(-1)), dim=-1)
    # Only select the first 1024 elements
    sliding_avg = sliding_avg[:, :, :1024]
    return sliding_avg

def plot_and_save(data, name):
    plt.figure()
    plt.plot(np.arange(2048),data)
    # plt.clf()
    plt.savefig('/home/wangxp/DataModel/img/'+str(name)+'.png')

def transform_data(data1):
    N = data1.shape[0]
    data2 = torch.zeros((N, 8, 1024))
    for i in range(8):
        data2[:, i, :] = data1[:, :, i*256:(i+1)*256].reshape(N, -1)
    return data2

def inverse_transform_data(data2):
    N = data2.shape[0]
    data1 = torch.zeros((N, 4, 2048))
    for i in range(8):
        data1[:, :, i*256:(i+1)*256] = data2[:, i, :].reshape(N, 4, -1)
    return data1

def positional_encoding(seq_len=8, d_model=1024):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    return PE

def create_mask_and_indices(data, k):
    N, T, _ = data.size()
    mask = torch.zeros(N, T, dtype=torch.bool)

    indices = []
    for n in range(N):
        idx = torch.randperm(T)[:k]
        mask[n, idx] = True
        indices.append(idx.tolist())

    return mask, indices

def create_mask_and_indices_and_complement(data, k):
    N, T, _ = data.size()
    mask = torch.zeros(N, T, dtype=torch.bool)

    indices = []
    complement_indices = []
    for n in range(N):
        idx = torch.randperm(T)[:k]
        mask[n, idx] = True
        indices.append(idx.tolist())

        complement_idx = torch.tensor([i for i in range(T) if i not in idx.tolist()])
        complement_indices.append(complement_idx.tolist())

    return mask, indices, complement_indices

def MAEData(x_train,NUM_MUSK,device):
    # x_train input
    x_recon = torch.tensor(x_train);x_recon=x_recon.to(device)
    key_mask, indices = create_mask_and_indices(x_train, NUM_MUSK);key_mask=key_mask.to(device);
    key_mask, indices,indices_r=create_mask_and_indices_and_complement(x_train, NUM_MUSK);key_mask=key_mask.to(device);
    bti = torch.arange(x_train.shape[0]).unsqueeze(1).expand(-1, NUM_MUSK);x_train[bti, indices, :]=1e-10;
    x_train += positional_encoding(seq_len=8,d_model=1024);
    bti2 = torch.arange(x_train.shape[0]).unsqueeze(1).expand(-1, 8-NUM_MUSK);
    x_train = x_train[bti2, indices_r, :];
    
    
    # # 创建一个新的tensor，其形状与x_train相同，所有元素初始化为1e-10
    # x_train_recon = torch.full(x_recon.shape, 1e-10, device=device)
    # # 使用indices_r和bti2将x_train的部分数据复制到新的tensor中的相应位置
    # for i in range(x_recon.shape[0]):
    #     x_train_recon[i, indices_r[i], :] = x_train[i, :len(indices_r[i]), :]
    return x_train,key_mask,indices,bti,indices_r,bti2


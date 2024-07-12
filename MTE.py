# %% [markdown]
# **Transformer for fingerprint localization, by Xiping in Apr-10-2024.**        
# - Use transformer or attention-based DL models to localize positions with CSI as fingerprint.
# - 2 scnearios: xh & sy; 3D localization; 4 Rx; Frequency point(F_p): 2048;
# 
# - Please name the experiments name 
# 

# %%
# Import
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from datetime import datetime
import random

from DataProcessingModule import Add_Gaussian_Noise, Standardization, sliding_average
from DataProcessingModule import transform_data, inverse_transform_data, positional_encoding,create_mask_and_indices
from DataProcessingModule import CustomLoss_L2, CustomLoss_L1,Normalization,scale_down,scale_up, CDF_save

from MTE_model import  ViT
from torch.utils.tensorboard import SummaryWriter
now = datetime.now()
# Create a tensorboard writer
log_dir = 'TestLog/' + now.strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir)


# %%
# CUDA_VISIBLE_DEVICES=1 
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL = 0
# SEED = 41
# np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# scenario_name = 'sy';# sy[19,9,9] w=[1,0.75,1], xh [21,7,9] w=[1.1, 1, 1]
scenario_name = 'sy_all';# sy_all[4, 73, 25, 15, 2, 2048] w=[0.25,0.25,0.5], xh_all [4, 81, 25, 13, 2, 2048] w=[0.275, 0.25, 0.25]
EM_index_ori = 'ori' # ori,1,2,3,4  
EM_index_compare = '4' # ori,1,2,3,4

# cube location (No use, but must exist to run the code)
ci_x = 30; ci_y = 12; ci_z = 8; x_range = 10; y_range = 10; z_range = 5; # sy
# ci_x = 40; ci_y = 12; ci_z = 5; x_range = 10; y_range = 10; z_range = 5; # xh

# Configuration
Noise_indicator = 1; sigma_dB= 1e-2; HeteroRatio = 0.1; # Electricity of CTF * sigma dB. defalt 1e-1 - 20 dB ; 0.316 - 10 dB, 0.05
abs_indicator = 1; dB_indicator = 1; STD_indicator = 1; TransformerSwitch = 1;
# x_range = 1; y_range = 1; z_range = 1;
BatchSize = 50;
epochs = 1000; 
NUM_MUSK = 0; # Set how many random sub-channels unavailable 0-8
DATA_PATH = 'sy_data.pt' # 'sy_data.pt' 'xh_data.pt' 
# %%
# Cut out a cube from full data
def locate_in_cube(data_full, ci_x, ci_y, ci_z, x_range = 1, y_range = 1, z_range = 1):
    data_cube = data_full[:, ci_x-x_range:ci_x+x_range+1, ci_y-y_range:ci_y+y_range+1, ci_z-z_range:ci_z+z_range+1, :, :] # [rx, ci_x, ci_y, ci_z, 2, fn]
    return data_cube

class CSI_Dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# Transform input_tensor to [ N,rx_N, 2048] and label to [N, 3]
def transform_tensor(input_tensor, Noise_indicator = False, abs_indicator = True, dB_indicator = True):
    rx_N, X, Y, Z, _, fp = input_tensor.shape
    N = X * Y * Z

    # Reshape the input tensor to [rx_N, N, 2048], 
    data = input_tensor.reshape(rx_N, N, 2, fp)
    if abs_indicator == True:
        data = torch.abs(torch.tensor(data[:,:,0,:] + 1j*data[:,:,1,:]))
    if dB_indicator == True:
        data = 20*torch.log10(data)
    # Create labels from [0,0,0] to [X,Y,Z]
    label = np.array([[x-x_range, y-y_range, z-z_range] for x in range(X) for y in range(Y) for z in range(Z)])
    # label = np.array([[x, y, z] for x in range(X) for y in range(Y) for z in range(Z)])

    # [N,rx_N, 2048]
    data = data.permute(1, 0, 2)
    return data, label



# %%
def train(model, optimizer, criterion, data_all_ori):
    train_data = data_all_ori
    ## model configuration
    total_loss = 0
    model.train() 
    for i, (x_train, y_train) in enumerate(train_data):
        if Noise_indicator : x_train =  Add_Gaussian_Noise(x_train, sigma_dB, HeteroRatio)# Add noise according to every Rx   
        x_train = sliding_average(sliding_average(x_train, 16),16); 
        if STD_indicator : x_train = Normalization(Standardization(x_train))
        if TransformerSwitch:
            x_train = transform_data(x_train) # to [N,8,1024]
            key_mask, indices = create_mask_and_indices(x_train, NUM_MUSK);key_mask=key_mask.to(device)
            bti = torch.arange(x_train.shape[0]).unsqueeze(1).expand(-1, NUM_MUSK);x_train[bti, indices, :]=1e-10;
            x_train += positional_encoding(seq_len=8,d_model=1024)
        x_train = x_train.float(); x_train = x_train.to(device); y_train = y_train.float(); y_train = y_train.to(device) ;
        optimizer.zero_grad()
        if TransformerSwitch: output = model(x_train, key_mask)
        else: output = model(x_train) # Train has float outputs
        if torch.isnan(output).any().item():
            print(y_train)
            break
        loss, var_x, var_y, var_z, num_rate, mean_x, mean_y, mean_z = criterion(output, y_train, scenario_name,device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Add gradient clipping
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_data)
    return average_loss

def evaluate(model, criterion, data_all_ori):

    
    eval_data = data_all_ori
    # Evaluate mode
    model.eval()
    total_loss = 0; total_success_num = 0
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(eval_data): 
            if Noise_indicator : x_val = Add_Gaussian_Noise(x_val, sigma_dB, HeteroRatio)# Add noise according to every Rx   
            x_val = sliding_average(sliding_average(x_val, 16),16); 
            if STD_indicator : x_val = Normalization(Standardization(x_val))
            if TransformerSwitch:
                x_val = transform_data(x_val)
                key_mask, indices = create_mask_and_indices(x_val, NUM_MUSK);key_mask= key_mask.to(device)
                bti = torch.arange(x_val.shape[0]).unsqueeze(1).expand(-1, NUM_MUSK);x_val[bti, indices, :]=1e-10;
                x_val += positional_encoding(seq_len=8,d_model=1024)

            x_val = x_val.float(); x_val = x_val.to(device);y_val = y_val.float(); y_val = y_val.to(device);
            if TransformerSwitch: output = torch.round(model(x_val,key_mask))
            loss, var_x, var_y, var_z, num_rate, mean_x, mean_y, mean_z = criterion(output, y_val, scenario_name,device)
            total_loss += loss.item()
            total_success_num += num_rate
    eval_score = total_loss / len(eval_data); success_rate = total_success_num / len(eval_data)
    if random.random() < 0.1: CDF_save(output, y_val, scenario_name,log_dir+'val')
    return eval_score, success_rate, var_x, var_y, var_z, mean_x, mean_y, mean_z

# %%
def main():
    #Load data
    # data_all_ori, data_all_compare = LoadData(scenario_name, EM_index_ori, EM_index_compare, device)
    data_cube = torch.load(DATA_PATH)
    print(data_cube.shape)
    X_data, Y_data = transform_tensor(data_cube)
    dataset = CSI_Dataset(X_data, Y_data);train_size = int(0.8 * len(dataset));test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_data = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
    eval_data = DataLoader(test_dataset, batch_size=X_data.shape[0], shuffle=True)
    
    model = ViT(d_model=1024, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1)
    model = model.to(device)
    criterion_train = CustomLoss_L2()  # Use mean square error as the loss function
    criterion_eval = CustomLoss_L2()
    lr = 0.0001 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
    best_score = float('inf');best_model_path = log_dir+'/best_model.pt';

    for epoch in range(1, epochs + 1):

        # training process
        total_loss = train(model, optimizer, criterion_train, train_data)
        writer.add_scalar('training loss', total_loss, epoch)

        # evaluation process
        eval_score, success_rate, var_x, var_y, var_z, mean_x, mean_y, mean_z = evaluate(
            model, criterion_eval, eval_data)
        writer.add_scalar('eval error', eval_score, epoch); 
        writer.add_scalar('success_rate', success_rate, epoch)
        print('Epoch: {}, Train_error: {:.3f}, Eval_error: {:.3f}, Success_rate: {:.5f}'.format(epoch,total_loss, eval_score, success_rate))
        print('Var_x: {:.3f}, Var_y: {:.3f}, Var_z: {:.3f}'.format(var_x, var_y, var_z))
        print('mean_x: {:.3f}, mean_y: {:.3f}, mean_z: {:.3f}'.format(mean_x, mean_y, mean_z))
        if eval_score < best_score:
            best_score = eval_score
            if SAVE_MODEL : torch.save(model.state_dict(), best_model_path)


if __name__ == "__main__":
    main()



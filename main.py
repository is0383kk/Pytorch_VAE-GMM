import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from tool import visualize_gmm

model_dir = "./model"
dir_name = "./model/debug" # debugフォルダに保存される
graph_dir = "./model/debug/graph" # 各種グラフの保存先
pth_dir = "./model/debug/pth" # VAEのパラメータの保存先

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
if not os.path.exists(pth_dir):
    os.mkdir(pth_dir)
if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)

parser = argparse.ArgumentParser(description='VAE-GMM MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda")


to_tensor_transforms = transforms.Compose([
    transforms.ToTensor()
])
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# データセット分割調整
trainval_dataset = datasets.MNIST('./../data', train=True, transform=transforms.ToTensor(), download=True)
n_samples = len(trainval_dataset) 
train_size = int(n_samples * 0.1) # 6000枚
#train_size = int(n_samples * 0.01) # 600枚
print(f"Number of training datasets :{train_size}")
subset1_indices = list(range(0,train_size)); subset2_indices = list(range(train_size,n_samples)) 
train_dataset = Subset(trainval_dataset, subset1_indices); val_dataset = Subset(trainval_dataset, subset2_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False) # 訓練用ローダ（１枚づつ）
all_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=False) # データセット総数分のローダ

import vae_module
import gmm_module



def train_model(mutual_iteration, dir_name, train_loader, all_loader):
    N = 1 # 試行回数（相互学習の回数とは別）
    ARI_list = np.zeros((mutual_iteration,N))
    for n in range(N):
        print(f"================={n+1}試行=================")
        for it in range(mutual_iteration):
            print(f"------------------相互学習{it+1}回目------------------")
            if it == 0: # VAE->GMMモデル
                gmm_mu = None; gmm_var = None
            
            # VAEモジュール
            x_d, label, loss_list = vae_module.train(
                iteration=it, gmm_mu=gmm_mu, gmm_var=gmm_var,
                epoch=100, train_loader=train_loader, all_loader=all_loader, 
                model_dir=dir_name
                )
            
            # GMMモジュール    
            gmm_mu, gmm_var, max_ARI = gmm_module.train(
                iteration=it, x_d=x_d,
                model_dir=dir_name,
                label=label, K=10, epoch=100, 
                )
            vae_module.plot_latent(iteration=it, all_loader=all_loader, model_dir=dir_name)
            ARI_list[it][n]=max_ARI
        
        print(f"ARI_mean :{np.mean(ARI_list, axis=1)}")

def plot_dist(load_iteration, decode_k, sample_num, dir_name):
    visualize_gmm(iteration=load_iteration, decode_k=decode_k, sample_num=sample_num, model_dir=dir_name)

def decode_from_gmm_param(iteration, decode_k, sample_num, model_dir="./vae_gmm"):
    vae_module.decode(iteration=iteration, decode_k=decode_k, sample_num=sample_num, model_dir=model_dir)

def main():
    train_model(mutual_iteration=10, dir_name=dir_name, train_loader=train_loader, all_loader=all_loader)
    load_iteration = 8 # 読み込むイテレーション.
    decode_k = 0 # 読み込むカテゴリ.
    #plot_dist(load_iteration=load_iteration, decode_k=decode_k, sample_num=32, dir_name=dir_name)
    #decode_from_gmm_param(iteration=load_iteration, decode_k=decode_k, sample_num=32, model_dir=dir_name)

if __name__=="__main__":
    main()

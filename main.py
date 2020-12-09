import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
import vae_module
import gmm_module

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--z_dim', type=int, default=20, metavar='N',
                    help='number of dimention of latent variable')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

#データセットの定義
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# データセット分割調整
trainval_dataset = datasets.MNIST('../../data', train=True, transform=transforms.ToTensor(), download=True)
n_samples = len(trainval_dataset) 
train_size = int(n_samples * 0.1) 
subset1_indices = list(range(0,train_size)) 
subset2_indices = list(range(train_size,n_samples)) 
train_dataset = Subset(trainval_dataset, subset1_indices)
val_dataset   = Subset(trainval_dataset, subset2_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
send_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=False)


def train_model(train_loader, send_data_loader, file_name="./model"): # VAE-GMMモデルを学習させる際に使用
    flag = True # 初回の学習時はGMMからガウス分布の\muを受け取らないためフラグを立てておく
    gmm_mu = None # GMMが推定した\mu.VAEの１回目の学習では使用しないためNoneを設定

    # 学習
    for i in range(10):
        print(f"----------相互学習{i+1}回目----------")
        vae_z = vae_module.train( gmm_mean=gmm_mu, epoch=100, first=flag, train_loader=train_loader, send_data_loader=send_data_loader, save_dir=file_name)
        gmm_pdz, gmm_mu = gmm_module.train( data=vae_z, K=10, num_itr=100, save_dir=file_name) # gdm_pdz:(600,10), gmm_mu:(600,20)
        gmm_pdz, gmm_mu, classes = gmm_module.test( data=vae_z, K=10, load_dir=file_name)
        vae_module.plot_latent(send_data_loader=send_data_loader, file_name=file_name, classes=classes)
        flag = False
    # 分類精度を測る
    #vae_z = vae_module.send_latent(send_data_loader=send_data_loader,load_dir=file_name)
    #gmm_pdz, gmm_mu = gmm_module.train( data=vae_z, K=10, num_itr=150, save_dir="vae_gmm")
    #gmm_pdz, gmm_mu, classes = gmm_module.test( data=vae_z, K=10, load_dir=file_name)
    #vae_module.plot_latent(send_data_loader=send_data_loader, load_dir=file_name, image_name="latent_space", classes=classes)

def inference(send_data_loader, file_name="./model"):
    # 分類精度を測る
    vae_z = vae_module.send_latent(send_data_loader=send_data_loader, load_dir=file_name)
    gmm_pdz, gmm_mu, classes = gmm_module.test( data=vae_z, K=10, load_dir=file_name)
    vae_module.plot_latent(send_data_loader=send_data_loader, file_name=file_name, classes=classes)


def main():
    train_model(train_loader=train_loader, send_data_loader=send_data_loader)
    #inference(send_data_loader=send_data_loader)

if __name__=="__main__":
    main()
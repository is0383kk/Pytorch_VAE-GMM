from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import numpy as np
import gmm_module
# 潜在変数の可視化
from sklearn.manifold import TSNE
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score as ARI


parser = argparse.ArgumentParser(description='VAE MNIST Example')
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

z_dim=20
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, z_dim)
        self.fc22 = nn.Linear(256, z_dim)
        self.fc3 = nn.Linear(z_dim, 256)
        self.fc4 = nn.Linear(256, 784)
        # 事前分布のパラメータを定義
        self.prior_var = nn.Parameter(torch.Tensor(1, z_dim).float().fill_(1.0))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, gmm_mean):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114       
        gmm_mean = nn.Parameter(gmm_mean)
        prior_mean = gmm_mean
        prior_mean.requires_grad = False
        prior_mean = prior_mean.expand_as(mu).to(device)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - z_dim)
        return BCE + KLD


def train(gmm_mean, epoch, first, train_loader, send_data_loader, save_dir="./vae_gmm"):
    prior_mean = torch.Tensor(len(train_loader), z_dim).float().fill_(0.0) # 最初のVAEの事前分布の\mu
    model = VAE().to(device)
    #loss_list = []
    #epoch_list = np.arange(epoch)
    if first!=True:
        print("前回の学習パラメータの読み込み")
        #model.load_state_dict(torch.load(save_dir))
    #model.load_state_dict(torch.load(save_dir))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("VAE内の学習メソッド")
    for it in range(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(data)
            if first==True: # 最初の学習
                loss = model.loss_function(recon_batch, data, mu, logvar, prior_mean[batch_idx])
            else: # ２回目以降の学習（GMMから正規分布のパラメータを受け取る）
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mean[batch_idx])
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            #if batch_idx % args.log_interval == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(data), len(train_loader.dataset),
            #        100. * batch_idx / len(train_loader),
            #        loss.item() / len(data)))
        if it == 0 or (it+1) % 25 == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
            it+1, train_loss / len(train_loader.dataset)))
        #loss_list.append((train_loss / len(train_loader.dataset).numpy()))
    #np.save('./loss_list.npy', np.array(loss_list))
    torch.save(model.state_dict(), save_dir+"/vae.pth")
    return send_latent(send_data_loader=send_data_loader, load_dir=save_dir)

def send_latent(send_data_loader, load_dir="./vae_gmm"): # gmmにvaeの潜在空間を送るメソッド
    model = VAE().to(device)
    model.load_state_dict(torch.load(load_dir+"/vae.pth"))
    model.eval()
    for batch_idx, (data, _) in enumerate(send_data_loader):
        data = data.to(device)
        recon_batch, mu, logvar, z = model(data)
        z = z.cpu()
    return z.detach().numpy()

def visualize(z, labels, file_name, acc=None):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    fig = plt.figure(figsize=(10,10))
    points = TSNE(n_components=2, random_state=0).fit_transform(z)
    for p, l in zip(points, labels):
        if acc is None:
            plt.title("Latent space", fontsize=24)
        else:
            plt.title("ACC="+str(acc)+":Latent space", fontsize=24)
        plt.xlabel("Latent space:xlabel", fontsize=21)
        plt.ylabel("Latent space:ylabel", fontsize=21)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
    plt.savefig(file_name+'./latent_space.png')
    print("画像保存完了")

def plot_latent(send_data_loader, file_name="./vae_gmm", classes=None):
    print("潜在変数可視化メソッド")
    model = VAE().to(device)
    model.load_state_dict(torch.load(file_name+"/vae.pth"))
    model.eval()
    for batch_idx, (data, label) in enumerate(send_data_loader):
        data = data.to(device)
        recon_batch, mu, logvar, z = model(data)
        z = z.cpu()
        if classes is not None:
            acc, result = gmm_module.calc_acc(classes, label.numpy())
            #acc = ARI(label.numpy(), classes)
            acc = round(acc,3)
            print(f"acc:{round(acc,3)}")
        visualize(z.detach().numpy(), label, file_name, acc=acc)
        break
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
import matplotlib.pyplot as plt
from tool import visualize_ls, sample, get_param



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
z_dim=12
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, z_dim)
        self.fc22 = nn.Linear(256, z_dim)
        self.fc3 = nn.Linear(z_dim, 256)
        self.fc4 = nn.Linear(256, 784)
        # 事前分布のパラメータN(0,I)で初期化
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
    def loss_function(self, recon_x, x, en_mu, en_logvar, gmm_mu, gmm_var, iteration):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        beta = 5.0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114 
        if iteration != 0: 
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.expand_as(en_mu).to(device)
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.expand_as(en_logvar).to(device)
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(en_logvar).to(device)
            
            var_division = en_logvar.exp() / prior_var # Σ_0 / Σ_1
            diff = en_mu - prior_mu # μ_１ - μ_0
            diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
            logvar_division = prior_logvar - en_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
            KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - z_dim)
        else:
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())
        return BCE + KLD

def train(iteration, gmm_mu, gmm_var, epoch, train_loader, all_loader, model_dir="./vae_gmm"):
    prior_mean = torch.Tensor(len(train_loader), z_dim).float().fill_(0.0) # 最初のVAEの事前分布の\mu
    model = VAE().to(device)
    print("VAE Training Start")
    #loss_list = []
    #epoch_list = np.arange(epoch)
    #model.load_state_dict(torch.load(save_dir+"/vae.pth"))
    #if first!=True:
        #print("前回の学習パラメータの読み込み")
        #model.load_state_dict(torch.load(save_dir+"/vae.pth"))
    
    #model.load_state_dict(torch.load(save_dir))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = np.zeros((epoch))
    for i in range(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(data)
            if iteration==0: # 最初の学習
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration)
            else:
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu[batch_idx], gmm_var[batch_idx], iteration=iteration)
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if i == 0 or (i+1) % 50 == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
            i+1, train_loss / len(train_loader.dataset)))
            
        loss_list[i] = train_loss / len(train_loader.dataset)
    #グラフ処理
    plt.figure()
    plt.plot(range(0,epoch), loss_list, color="blue", label="loss")
    plt.xlabel('epoch')
    plt.ylabel('VAE_loss')
    plt.savefig(model_dir+'/graph/vae_loss_'+str(iteration)+'.png')
    plt.close()
    #np.save('./loss_list.npy', np.array(loss_list))
    torch.save(model.state_dict(), model_dir+"/pth/vae_"+str(iteration)+".pth")
    x_d, label = send_all_z(iteration=iteration, all_loader=all_loader, model_dir=model_dir)
    return x_d, label, loss_list
    

def decode(iteration, decode_k, sample_num, model_dir="./vae_gmm"):
    model = VAE().to(device)
    model.load_state_dict(torch.load(str(model_dir)+"/pth/vae_"+str(iteration)+".pth"))
    model.eval()
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration, model_dir=model_dir)
    manual_sample, random_sample = sample(iteration=iteration, z_dim=z_dim, 
                     mu_gmm=mu_gmm_kd, lambda_gmm=lambda_gmm_kdd, 
                     sample_num=sample_num, sample_k=decode_k, model_dir=model_dir
                     )
    sample_d = manual_sample
    sample_d = torch.from_numpy(sample_d.astype(np.float32)).clone()
    print(sample_d)
    with torch.no_grad():
        sample_d = sample_d.to(device)
        #sample_d = torch.from_numpy(sample_d).to(device)
        sample_d = model.decode(sample_d).cpu()
        save_image(sample_d.view(sample_num, 1, 28, 28),'recon/manual_'+str(decode_k)+'.png')
    

def plot_latent(iteration, all_loader, model_dir="./vae_gmm"): # VAEの潜在空間を可視化するメソッド
    print("Plot latent space")
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae_"+str(iteration)+".pth"))
    model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        data = data.to(device)
        recon_batch, mu, logvar, z = model(data)
        z = z.cpu()
        #if classes is not None:
        #    ari, result = calc_ari(classes, label.numpy())
        #    print(f"ari:{round(ari,3)}")
        #    ari = round(ari,3)
        #np.savetxt('z5000.txt', z.detach().numpy())
        #np.savetxt('label5000.txt', label.detach().numpy())
        visualize_ls(iteration, z.detach().numpy(), label, model_dir)
        break

def test(epoch):
    model = VAE().to(device)
    model.load_state_dict(torch.load(file_name+"/vae.pth"))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, args.category)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 18)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'image/recon_' + str(epoch) + '.png', nrow=n)

def send_all_z(iteration, all_loader, model_dir="./vae_gmm"): # gmmにvaeの潜在空間を送るメソッド
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae_"+str(iteration)+".pth"))
    model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        data = data.to(device)
        recon_batch, mu, logvar, z = model(data)
        z = z.cpu()
        label = label.cpu()
    return z.detach().numpy(), label.detach().numpy()
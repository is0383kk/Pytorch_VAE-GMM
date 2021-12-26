import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from random import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def sample(iteration, x_dim, mu_gmm, lambda_gmm, sample_num, sample_k, model_dir="./vae_gmm"):
    sigma = 0.1
    sigma_kdd = sigma * np.identity(x_dim, dtype=float)
    """
    Sample the random variable that will be the input to the VAE decoder 
    from the posterior distribution estimated by the GMM
    """
    manual_sample = np.random.multivariate_normal(mean=mu_gmm[sample_k], cov=sigma_kdd, size=sample_num)
    random_sample = np.random.multivariate_normal(mean=mu_gmm[sample_k], cov=np.linalg.inv(lambda_gmm[sample_k]), size=sample_num)

    return manual_sample, random_sample

def visualize_gmm(iteration, decode_k, sample_num, model_dir="./vae_gmm"):
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration=iteration, model_dir=model_dir)
    manual_sample, random_sample = sample(iteration=iteration, x_dim=12, 
                                          mu_gmm=mu_gmm_kd, lambda_gmm=lambda_gmm_kdd, 
                                          sample_k=decode_k, sample_num=sample_num, model_dir=model_dir
                                          )
    mu_gmm2d_kd = np.zeros((10,2)) # mu 2 dimention
    lambda_gmm2d_kdd = np.zeros((10,2,2)) # lambda 2 dimention
    for k in range(10):
        mu_gmm2d_kd[k] = mu_gmm_kd[k][:2]
        for dim1 in range(2):
            for dim2 in range(2):
                lambda_gmm2d_kdd[k][dim1][dim2] = lambda_gmm_kdd[k][dim1][dim2]
    
    x_1_line = np.linspace(
    np.min(mu_gmm_kd[:, 0] - 0.5 * np.sqrt(lambda_gmm_kdd[:, 0, 0])), 
    np.max(mu_gmm_kd[:, 0] + 0.5 * np.sqrt(lambda_gmm_kdd[:, 0, 0])), 
    num=900
    )
    
    x_2_line = np.linspace(
    np.min(mu_gmm_kd[:, 1] - 0.5 * np.sqrt(lambda_gmm_kdd[:, 1, 1])), 
    np.max(mu_gmm_kd[:, 1] + 0.5 * np.sqrt(lambda_gmm_kdd[:, 1, 1])), 
    num=900
    )
    
    x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)
    x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
    x_dim = x_1_grid.shape

    
    res_density_k = 0
    tmp_density_k = multivariate_normal.pdf(x=x_point, mean=mu_gmm_kd[decode_k][:2], cov=np.linalg.inv(lambda_gmm2d_kdd[decode_k]))
    res_density_k += tmp_density_k * pi_gmm_k[0]

    plt.figure(figsize=(12, 9))
    plt.scatter(x=manual_sample[:, 0], y=manual_sample[:, 1], label='cluster:' + str(k + 1)) 
    plt.scatter(x=mu_gmm2d_kd[:, 0], y=mu_gmm2d_kd[:, 1], color='red', s=100, marker='x') 
    plt.contour(x_1_grid, x_2_grid, res_density_k.reshape(x_dim), alpha=0.5, linestyles='dashed') 
    #plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), linestyles='--') # 真の分布
    plt.suptitle('Gaussian Mixture Model', fontsize=20)
    plt.title('Number of sample='+str(len(manual_sample))+', K='+str(decode_k))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.colorbar()
    plt.savefig(model_dir+'/graph/gause_I'+str(iteration)+'k'+str(decode_k)+'.png')
    plt.show()
    plt.close()

def get_param(iteration, model_dir="./vae_gmm"):
    mu_gmm_kd = np.load(model_dir+"/npy/mu_"+str(iteration)+".npy") 
    lambda_gmm_kdd = np.load(model_dir+"/npy/lambda_"+str(iteration)+".npy") 
    pi_gmm_k = np.load(model_dir+"/npy/pi_"+str(iteration)+".npy") 
    
    return mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k

def visualize_ls(iteration, z, labels, save_dir):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    #points = PCA(n_components=2, random_state=0).fit_transform(z)
    points = TSNE(n_components=2, random_state=0).fit_transform(z)
    plt.figure(figsize=(10,10))
    for p, l in zip(points, labels):
        plt.title("Latent space on VAE")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
    plt.savefig(save_dir+'/graph/z_'+str(iteration)+'.png')
    plt.close()

def calc_acc( results, correct ):
    K = np.max(results)+1     # Number of category
    D = len(results)          # Number of data points
    max_acc = 0               # Max acc
    changed = True
    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros( D )

                for n in range(D):
                    if results[n]==i: tmp_result[n]=j
                    elif results[n]==j: tmp_result[n]=i
                    else: tmp_result[n] = results[n]

                # Caluculate acc
                acc = (tmp_result==correct).sum()/float(D)

                if acc > max_acc:
                    max_acc = acc
                    results = tmp_result
                    changed = True

    return max_acc, results

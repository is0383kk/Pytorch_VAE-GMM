import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from scipy.stats import multivariate_normal, wishart, dirichlet



def sample(iteration, z_dim, mu_gmm, lambda_gmm, sample_num, sample_k, model_dir="./vae_gmm"):
    """
    iteration:サンプルするモデルのイテレーション
    z_dim:VAEの潜在変数の次元数（＝GMMの観測の次元数）
    mu_gmm:GMMの推定した平均パラメータ
    lambda_gmm:GMMの推定した精度行列パラメータ
    sample_num:サンプル数
    sample_k:サンプルする際のK
    """
    sigma = 0.001
    sigma_kdd = sigma * np.identity(z_dim, dtype=float) # 対角成分がsigmaでそれ以外は0の分散共分散行列
    # サンプリングデータを生成
    #x_nd = np.random.multivariate_normal(mean=mu_gmm[k], cov=lambda_gmm[k], size=sample_num)
    manual_sample = np.random.multivariate_normal(mean=mu_gmm[sample_k], cov=sigma_kdd, size=sample_num)
    random_sample = np.random.multivariate_normal(mean=mu_gmm[sample_k], cov=np.linalg.inv(lambda_gmm[sample_k]), size=sample_num)

    return manual_sample, random_sample

def visualize_gmm(iteration, decode_k, sample_num, model_dir="./vae_gmm"):
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration=iteration, model_dir=model_dir)
    manual_sample, random_sample = sample(iteration=iteration, z_dim=12, 
                                          mu_gmm=mu_gmm_kd, lambda_gmm=lambda_gmm_kdd, 
                                          sample_k=decode_k, sample_num=sample_num, model_dir=model_dir
                                          )
    mu_gmm2d_kd = np.zeros((10,2)) # mu 2次元化
    lambda_gmm2d_kdd = np.zeros((10,2,2)) # lambda 2次元化
    for k in range(10):
        mu_gmm2d_kd[k] = mu_gmm_kd[k][:2]
        for dim1 in range(2):
            for dim2 in range(2):
                lambda_gmm2d_kdd[k][dim1][dim2] = lambda_gmm_kdd[k][dim1][dim2]
    # 作図用のx軸のxの値を作成
    x_1_line = np.linspace(
    np.min(mu_gmm_kd[:, 0] - 0.5 * np.sqrt(lambda_gmm_kdd[:, 0, 0])), 
    np.max(mu_gmm_kd[:, 0] + 0.5 * np.sqrt(lambda_gmm_kdd[:, 0, 0])), 
    num=900
    )
    # 作図用のy軸のxの値を作成
    x_2_line = np.linspace(
    np.min(mu_gmm_kd[:, 1] - 0.5 * np.sqrt(lambda_gmm_kdd[:, 1, 1])), 
    np.max(mu_gmm_kd[:, 1] + 0.5 * np.sqrt(lambda_gmm_kdd[:, 1, 1])), 
    num=900
    )
    # 作図用の格子状の点を作成
    x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)
    # 作図用のxの点を作成
    x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
    # 作図用に各次元の要素数を保存
    x_dim = x_1_grid.shape

    # 最後にサンプルしたパラメータによる混合分布を計算
    res_density_k = 0
    # クラスタkの分布の確率密度を計算
    tmp_density_k = multivariate_normal.pdf(x=x_point, mean=mu_gmm_kd[decode_k][:2], cov=np.linalg.inv(lambda_gmm2d_kdd[decode_k]))
    # K個の分布の加重平均を計算
    res_density_k += tmp_density_k * pi_gmm_k[0]

    # 観測データの散布図を作成
    plt.figure(figsize=(12, 9))
    plt.scatter(x=manual_sample[:, 0], y=manual_sample[:, 1], label='cluster:' + str(k + 1)) # 観測データ
    plt.scatter(x=mu_gmm2d_kd[:, 0], y=mu_gmm2d_kd[:, 1], color='red', s=100, marker='x') # 事後分布の平均
    plt.contour(x_1_grid, x_2_grid, res_density_k.reshape(x_dim), alpha=0.5, linestyles='dashed') # K=0の等高線
    #plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), linestyles='--') # 真の分布
    plt.suptitle('Gaussian Mixture Model', fontsize=20)
    plt.title('Number of sample='+str(len(manual_sample))+', K='+str(decode_k))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.colorbar()
    plt.savefig(model_dir+'/graph/gause_'+str(iteration)+'k_'+str(decode_k)+'.png')
    plt.show()
    plt.close()

def get_param(iteration, model_dir="./vae_gmm"):
    mu_gmm_kd = np.load(model_dir+"/mu_"+str(iteration)+".npy") # GMMの平均パラメータ
    lambda_gmm_kdd = np.load(model_dir+"/lambda_"+str(iteration)+".npy") # GMMの逆共分散行列パラメータ
    pi_gmm_k = np.load(model_dir+"/pi_"+str(iteration)+".npy") # GMMの混合比
    
    return mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k

def visualize_ls(iteration, z, labels, save_dir):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    #print("潜在変数の可視化")
    #points = PCA(n_components=2, random_state=0).fit_transform(z)
    points = TSNE(n_components=2, random_state=0).fit_transform(z)
    plt.figure(figsize=(10,10))
    for p, l in zip(points, labels):
        plt.title("Latent space", fontsize=24)
        plt.xlabel("Latent space:xlabel", fontsize=21)
        plt.ylabel("Latent space:ylabel", fontsize=21)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
    plt.savefig(save_dir+'/graph/z_'+str(iteration)+'.png')
    plt.close()
    #print("画像保存完了")

def calc_ari( results, correct ):
    K = np.max(results)+1     # カテゴリ数
    D = len(results)          # データ数
    max_ari = 0               # 精度の最大値
    changed = True            # 変化したかどうか

    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros( D )

                # iとjを入れ替える
                for n in range(D):
                    if results[n]==i: tmp_result[n]=j
                    elif results[n]==j: tmp_result[n]=i
                    else: tmp_result[n] = results[n]

                # 精度を計算
                ari = (tmp_result==correct).sum()/float(D)

                # 精度が高くなって入れば保存
                if ari > max_ari:
                    max_ari = acc
                    results = tmp_result
                    changed = True

    return max_ari, results

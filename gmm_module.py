# 実装の参考元：https://www.anarchive-beta.com/entry/2020/11/28/210948
# ギブスサンプリング導出の参考元:https://www.anarchive-beta.com/entry/2020/11/18/182943
import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet 
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from tool import calc_ari
import torch


def train(iteration, x_d, label, K, epoch=100, model_dir="vae_gmm"):
    print("GMM Training Start")
    D = len(x_d) # データ総数
    dim = len(x_d[0]) # 次元数を設定:(固定)

    #事前分布のパラメータ
    beta = 0.8; m_d = np.repeat(0.0, dim) # muの事前分布のパラメータを指定
    w_dd = np.identity(dim) * 0.55; nu = dim # lambdaの事前分布のパラメータを指定
    alpha_k = np.repeat(0.3, K) # piの事前分布のパラメータを指定

    #\mu, \lambda, \piの初期値を決定
    # 観測モデルのパラメータをサンプル
    mu_kd = np.empty((K, dim)); lambda_kdd = np.empty((K, dim, dim))
    for k in range(K):
        # クラスタkの精度行列をサンプル
        lambda_kdd[k] = wishart.rvs(df=nu, scale=w_dd, size=1)
        # クラスタkの平均をサンプル
        mu_kd[k] = np.random.multivariate_normal(mean=m_d, cov=np.linalg.inv(beta * lambda_kdd[k])).flatten()
    pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten() # 混合比率をサンプル

    # パラメータを初期化
    eta_dk = np.zeros((D, K))
    z_dk = np.zeros((D, K))
    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, dim))
    w_hat_kdd = np.zeros((K, dim, dim))
    nu_hat_k = np.zeros(K)
    alpha_hat_k = np.zeros(K)

    # パラメータと指標の推移
    trace_s_in = [np.repeat(np.nan, D)]
    trace_mu_ikd = [mu_kd.copy()]
    trace_lambda_ikdd = [lambda_kdd.copy()]
    trace_pi_ik = [pi_k.copy()]
    trace_beta_ik = [np.repeat(beta, K)]
    trace_m_ikd = [np.repeat(m_d.reshape((1, dim)), K, axis=0)]
    trace_w_ikdd = [np.repeat(w_dd.reshape((1, dim, dim)), K, axis=0)]
    trace_nu_ik = [np.repeat(nu, K)]
    trace_alpha_ik = [alpha_k.copy()]
    ARI = np.zeros((epoch))
    max_ARI = 0 # イテレーション内の最高ARIを記録
    
    # ギブスサンプリング
    for i in range(epoch):
        pred_label = []
        
        # zのサンプリング
        # 潜在変数の事後分布のパラメータを計算:式(4.94)
        for k in range(K):
            tmp_eta_n = np.diag(
                -0.5 * (x_d - mu_kd[k]).dot(lambda_kdd[k]).dot((x_d - mu_kd[k]).T)
            ).copy() # (何故か書き替え禁止になるのを防ぐためのcopy())
            tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
            tmp_eta_n += np.log(pi_k[k] + 1e-7)
            eta_dk[:, k] = np.exp(tmp_eta_n)
        eta_dk /= np.sum(eta_dk, axis=1, keepdims=True) # 正規化
        
        # 潜在変数をサンプル：式(4.93)
        for d in range(D):
            z_dk[d] = np.random.multinomial(n=1, pvals=eta_dk[d], size=1).flatten()
            pred_label.append(np.argmax(z_dk[d]))
            
        #\mu, \lambdaのサンプリング
        # 観測モデルのパラメータをサンプル
        for k in range(K):
            
            # muの事後分布のパラメータを計算：式(4.99)
            beta_hat_k[k] = np.sum(z_dk[:, k]) + beta
            m_hat_kd[k] = np.sum(z_dk[:, k] * x_d.T, axis=1)
            m_hat_kd[k] += beta * m_d
            m_hat_kd[k] /= beta_hat_k[k]
            
            # lambdaの事後分布のパラメータを計算：式(4.103)
            tmp_w_dd = np.dot((z_dk[:, k] * x_d.T), x_d)
            tmp_w_dd += beta * np.dot(m_d.reshape(dim, 1), m_d.reshape(1, dim))
            tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(dim, 1), m_hat_kd[k].reshape(1, dim))
            tmp_w_dd += np.linalg.inv(w_dd)
            w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
            nu_hat_k[k] = np.sum(z_dk[:, k]) + nu
            
            # lambdaをサンプル：式(4.102)
            lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
            
            
            # muをサンプル：式(4.98)
            mu_kd[k] = np.random.multivariate_normal(
                mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1
            ).flatten()

        # \pi のサンプリング
        # 混合比率のパラメータを計算：式(4.45)
        alpha_hat_k = np.sum(z_dk, axis=0) + alpha_k
        
        # piをサンプル：式(4.44)
        pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
        
        ARI[i] = np.round(calc_ari(pred_label,label)[0],3)
        # ARI最大時のmu,lambdaを格納
        if max_ARI <= ARI[i]:
            max_ARI = ARI[i]
            
            
        
        if i == 0 or (i+1) % 50 == 0:
            print(f"====> Epoch: {i+1}, ARI: {ARI[i]}, MaxARI: {max_ARI}")
        
        # 値を記録
        _, z_n = np.where(z_dk == 1)
        trace_s_in.append(z_n.copy())
        trace_mu_ikd.append(mu_kd.copy())
        trace_lambda_ikdd.append(lambda_kdd.copy())
        trace_pi_ik.append(pi_k.copy())
        trace_beta_ik.append(beta_hat_k.copy())
        trace_m_ikd.append(m_hat_kd.copy())
        trace_w_ikdd.append(w_hat_kdd.copy())
        trace_nu_ik.append(nu_hat_k.copy())
        trace_alpha_ik.append(alpha_hat_k.copy())

    #print(f"lambda_kdd: {lambda_kdd},{lambda_kdd.shape}")
    #np.savetxt(str(save_dir)+'/pi_ik.txt', trace_pi_ik)
    mu_d = np.zeros((D,dim)) # GMMの平均パラメータ
    var_d = np.zeros((D,dim)) # GMMのLambdaの対角成分（分散）
    for d in range(D):
        #print(f"-------x_{d}:カテゴリ={np.argmax(z_dk[d])}")
        #var_d[d] = np.sqrt(np.diag(np.linalg.inv(lambda_kdd[pred_label[d]])))
        var_d[d] = np.diag(np.linalg.inv(lambda_kdd[pred_label[d]]))
        #var_d[d] = np.diag(lambda_kdd[pred_label[d]])
        mu_d[d] = mu_kd[pred_label[d]]
    plot(iteration=iteration, epoch=epoch, K=K, trace_pi_ik=trace_pi_ik, ari=ARI, model_dir=model_dir)
    np.save(model_dir+'/mu_'+str(iteration)+'.npy', mu_kd)
    np.save(model_dir+'/lambda_'+str(iteration)+'.npy', lambda_kdd)
    np.save(model_dir+'/pi_'+str(iteration)+'.npy', pi_k)

    return torch.from_numpy(mu_d), torch.from_numpy(var_d), max_ARI



def plot(iteration, epoch, K, trace_pi_ik , ari, model_dir):
    #グラフ処理
    plt.figure()
    plt.plot(range(0,epoch), ari, marker="None")
    plt.xlabel('iteration')
    plt.ylabel('ARI')
    plt.savefig(model_dir+'/graph/ari_'+str(iteration)+'.png')
    plt.close()
    # piの推移を作図
    #pi_truth_k = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    plt.figure(figsize=(12, 9))
    #plt.hlines(y=pi_truth_k, xmin=0, xmax=epoch + 1, label='true val', color='red', linestyles='--') # 真の値
    for k in range(K):
        plt.plot(np.arange(epoch + 1), np.array(trace_pi_ik)[:, k], label='pi_k=' + str(k + 1))
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.suptitle('pi', fontsize=20)
    plt.title('$\pi$', loc='left')
    plt.legend() # 凡例
    plt.grid() # グリッド線
    plt.savefig(model_dir+'/graph/pi_'+str(iteration)+'.png')
    plt.close()
    #plt.show()

"""
# muの推移を作図
plt.figure(figsize=(12, 9))
#plt.hlines(y=mu_truth_kd, xmin=0, xmax=epoch + 1, label='true val', color='red', linestyles='--') # 真の値
for k in range(K):
    for d in range(dim):
        plt.plot(np.arange(epoch+1), np.array(trace_mu_ikd)[:, k, d], 
                 label='k=' + str(k + 1) + ', d=' + str(d + 1))
plt.xlabel('epoch')
plt.ylabel('value')
plt.suptitle('Gibbs Sampling', fontsize=20)
plt.title('$\mu$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.savefig('mu.png')
plt.show()

plt.plot(range(0,epoch),ARI,marker="None")
plt.xlabel('epoch')
plt.ylabel('ARI')
plt.savefig("ari.png")
#plt.show()
"""

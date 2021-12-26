import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet 
import matplotlib.pyplot as plt
from tool import calc_acc
import torch


def train(iteration, x_d, label, K, epoch=100, model_dir="vae_gmm"):
    print("GMM Training Start")
    D = len(x_d) # number of data
    dim = len(x_d[0]) # dimention of latent variable of VAE

    # hyper parameter
    beta = 1.0; m_d = np.repeat(0.0, dim) 
    w_dd = np.identity(dim) * 0.55; nu = dim 
    alpha_k = np.repeat(0.3, K) 

    # initialized \mu, \lambda, \pi (sampling from prior)
    mu_kd = np.empty((K, dim)); lambda_kdd = np.empty((K, dim, dim))
    for k in range(K):
        lambda_kdd[k] = wishart.rvs(df=nu, scale=w_dd, size=1)
        mu_kd[k] = np.random.multivariate_normal(mean=m_d, cov=np.linalg.inv(beta * lambda_kdd[k])).flatten()
    pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten() 

    # initialize parameter
    eta_dk = np.zeros((D, K))
    z_dk = np.zeros((D, K))
    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, dim))
    w_hat_kdd = np.zeros((K, dim, dim))
    nu_hat_k = np.zeros(K)
    alpha_hat_k = np.zeros(K)

    ACC = np.zeros((epoch)) # Accuracy
    max_ACC = 0 
    
    # gibbssampling of gmm
    for i in range(epoch):
        pred_label = []
        
        for k in range(K):
            tmp_eta_n = np.diag(-0.5 * (x_d - mu_kd[k]).dot(lambda_kdd[k]).dot((x_d - mu_kd[k]).T)).copy()
            tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
            tmp_eta_n += np.log(pi_k[k] + 1e-7)
            eta_dk[:, k] = np.exp(tmp_eta_n)
        eta_dk /= np.sum(eta_dk, axis=1, keepdims=True)
        
        # sampling z
        for d in range(D):
            z_dk[d] = np.random.multinomial(n=1, pvals=eta_dk[d], size=1).flatten()
            pred_label.append(np.argmax(z_dk[d]))

        for k in range(K):
            beta_hat_k[k] = np.sum(z_dk[:, k]) + beta
            m_hat_kd[k] = np.sum(z_dk[:, k] * x_d.T, axis=1)
            m_hat_kd[k] += beta * m_d
            m_hat_kd[k] /= beta_hat_k[k]

            tmp_w_dd = np.dot((z_dk[:, k] * x_d.T), x_d)
            tmp_w_dd += beta * np.dot(m_d.reshape(dim, 1), m_d.reshape(1, dim))
            tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(dim, 1), m_hat_kd[k].reshape(1, dim))
            tmp_w_dd += np.linalg.inv(w_dd)
            w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
            nu_hat_k[k] = np.sum(z_dk[:, k]) + nu
            
            # sampling \lambda
            lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
            # sampling \mu
            mu_kd[k] = np.random.multivariate_normal(mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1).flatten()

        # sampling \pi
        alpha_hat_k = np.sum(z_dk, axis=0) + alpha_k        
        pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
        
        ACC[i] = np.round(calc_acc(pred_label,label)[0],3)
        
        if max_ACC <= ACC[i]:
            max_ACC = ACC[i]
        
        if i == 0 or (i+1) % 20 == 0 or i == (epoch-1):
            print(f"====> Epoch: {i+1}, Accuracy: {ACC[i]}, Max Accuracy: {max_ACC}")
        

    mu_d = np.zeros((D,dim))
    var_d = np.zeros((D,dim))
    for d in range(D):
        var_d[d] = np.diag(np.linalg.inv(lambda_kdd[pred_label[d]]))
        mu_d[d] = mu_kd[pred_label[d]]
    plot(iteration=iteration, epoch=epoch, K=K, acc=ACC, model_dir=model_dir)
    np.save(model_dir+'/npy/mu_'+str(iteration)+'.npy', mu_kd)
    np.save(model_dir+'/npy/lambda_'+str(iteration)+'.npy', lambda_kdd)
    np.save(model_dir+'/npy/pi_'+str(iteration)+'.npy', pi_k)

    return torch.from_numpy(mu_d), torch.from_numpy(var_d), max_ACC



def plot(iteration, epoch, K, acc, model_dir):
    plt.figure()
    plt.plot(range(0,epoch), acc, marker="None")
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.savefig(model_dir+'/graph/acc_'+str(iteration)+'.png')
    plt.close()
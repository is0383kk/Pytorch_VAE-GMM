# Variational Auto-Encoder(VAE)+Gaussian mixture model(GMM)
Implementation of a model to make VAE and GMM learn from each other.   
VAE sends x_d to GMM  
GNM sends the parameters(\mu,\Lambda^{-1}) of the Gaussian distribution to VAE  
This idea of integrating probability models is based on [this paper](https://arxiv.org/abs/1910.08918)   
This is a Graphical Model of VAE-GMM model:  

<div>
	<img src='/image/model.png' height="420px"><img src='/image/variable_define.png' width="420px">
</div>

## Changes with and without mutual learning  
### Latent space on VAE  
Left : without mutual learning・Right : with mutual learning  
<div>
	<img src='/image/z_tsne_0.png' width="380px"><img src='/image/z_tsne_1.png' width="380px">
</div>

### ELBO of VAE  
Red line is ELBO before mutual learning  
Blue line is ELBO after cross-learning  
(In general, the higher the ELBO, the better)  
<div>
	<img src='/image/vae_loss_1.png' height="380px">
</div>

### Clustering performance (in GMM)  
Results of clustering performance by accuracy(Addresses clustering performance in GMM within VAE+GMM)  
Left : without mutual learning・Right : with mutual learning  
<div>
	<img src='/image/acc_0.png' height="380px">
	<img src='/image/acc_1.png' height="380px">
</div>



## Image reconstruction from Gaussian distribution parameters estimated by GMM using VAE decoder  
Figure when a random variable is sampled from the Gaussian distribution of a particular category estimated by GMM（the diagonal component of the covariance matrix is set manually）  
The "X" marks are mean parameter for each category  
Blue circles are sampled random variables  
<div>
	<img src='/image/gause_8k_0.png' width="480px">
</div>
Reconstructed image of the sampled random variable input to the VAE decoder:
<div>
	<img src='/image/manual_0.png' width="380px">
</div>  

# Special Thanks  

[【Python】4.4.2：ガウス混合モデルにおける推論：ギブスサンプリング【緑ベイズ入門のノート】](https://www.anarchive-beta.com/entry/2020/11/28/210948)

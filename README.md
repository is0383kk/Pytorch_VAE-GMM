# Variational Auto-Encoder(VAE)+Gaussian mixture model(GMM)
Implementation of a model to make VAE and GMM learn from each other.   
VAE sends x_d to GMM  
GNM sends mean and variance parameters of the Gaussian distribution to VAE  
This idea of integrating probability models is based on [this paper](https://arxiv.org/abs/1910.08918)   
This is a Graphical Model of VAE+GMM model:  

<div>
	<img src='/image/model.png' height="420px"><img src='/image/variable_define.png' width="420px">
</div>

# How to run
You can train the VAE+GMM model by running `main.py`


# Changes with and without mutual learning  
## Latent space on VAE  
Left : without mutual learning・Right : with mutual learning  
<div>
	<img src='/image/z_tsne_0.png' width="380px"><img src='/image/z_tsne_1.png' width="380px">
</div>


## ELBO of VAE  
Red line is ELBO before mutual learning, Blue line is ELBO after mutual learning  
Vertical axis is training iteration of VAE, Horizontal axis is ELBO of VAE  
(In general, the higher the ELBO, the better)  
<div>
	<img src='/image/vae_loss_1.png' height="380px">
</div>

## Clustering performance (in GMM)  
Results of clustering performance by accuracy(Addresses clustering performance in GMM within VAE+GMM)  
Left : without mutual learning・Right : with mutual learning  
Vertical axis is training iteration of GMM, Horizontal axis is accuracy  
<div>
	<img src='/image/acc_0.png' height="380px"><img src='/image/acc_1.png' height="380px">
</div>



# Image reconstruction from Gaussian distribution parameters estimated by GMM using VAE decoder  
GMM performs clustering on latent variables of VAE. 
By sampling random variables from posterior distribution estimated by GMM and using them as input to VAE decoder, the image can be reconstructed.  
  
"x" represents the mean parameter of the normal distribution for each cluster.  
In this example, a random variable is sampled from a Gaussian distribution with K=1.
<div>
	<img src='/image/gause_I0k1.png' width="480px">
</div>
Reconstructed image of the sampled random variable input to the VAE decoder:
<div>
	<img src='/image/manual_1.png' width="380px">
</div>  


# Special Thanks  

[【Python】4.4.2：ガウス混合モデルにおける推論：ギブスサンプリング【緑ベイズ入門のノート】](https://www.anarchive-beta.com/entry/2020/11/28/210948)

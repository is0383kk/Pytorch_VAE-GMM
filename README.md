# Variational Auto-Encoder(VAE)+Gaussian mixture model(GMM)
Implementation of a model to make VAE and GMM learn from each other.   
VAE sends x_d to GMM  
GNM sends the parameters(\mu,\Lambda^{-1}) of the Gaussian distribution to VAE  
This is a Graphical Model of VAE-GMM model:  

<div>
	<img src='/image/model.png' height="420px">
</div>
The definition of each variable is as follows:
<div>
	<img src='/image/variable_define.png' width="380px">
</div>
Latent space of VAE before mutual learning:
<div>
	<img src='/image/latent_space_vae.png' width="380px">
</div>

Latent space of VAE after mutual learning:
<div>
	<img src='/image/latent_space_vaegmm.png' width="380px">
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

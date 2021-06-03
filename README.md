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
Reconstructed image when mean parameter of the Gaussian distribution of a particular category estimated by GMM is used as input to the decoder of VAE:
<div>
	<img src='/image/manual_0.png' width="380px">
</div>

# Variational Auto-Encoder(VAE)+Gaussian mixture model(GMM)
Implementation of a model to make VAE and GMM learn from each other  
VAE sends the latent space to GMM  
GNM sends the parameters of the Gaussian distribution to VAE  
This is a Graphical Model of VAE-GMM model:  

<div>
	<img src='/image/model.png' height="420px">
</div>
The definition of each variable is as follows:
<div>
	<img src='/image/variable_define.png' height="420px">
</div>
Latent space of VAE before mutual learning:
<div>
	<img src='/image/latent_space_vae.png' height="420px">
</div>

Latent space of VAE after mutual learning:
<div>
	<img src='/image/latent_space_vaegmm.png' height="420px">
</div>

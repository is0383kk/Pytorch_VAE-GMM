# Variational Auto-Encoder(VAE)+Gaussian mixture model(GMM)
Implementation of mutual learning model between VAE and GMM.  
This idea of integrating probability models is based on this paper: [Neuro-SERKET: Development of Integrative Cognitive System through the Composition of Deep Probabilistic Generative Models](https://arxiv.org/abs/1910.08918).  
Symbol Emergence in Robotics tool KIT（SERKET） is a framework that allows integration and partitioning of probabilistic generative models.  

This is a Graphical Model of VAE+GMM model:  

<div>
	<img src='/image/model.png' height="420px"><img src='/image/variable_define.png' width="420px">
</div>

VAE and GMM share the latent variable x.  
x is a variable that follows a multivariate normal distribution and is estimated by VAE.  

The training will be conducted in the following sequence.  
1. VAE estimates latent variable（x） and sends latent variables（x） to GMM.   
2. GMM clusters latent variables（x） sent from VAE and sends mean and variance parameters of the Gaussian distribution to VAE.  
3. Return to 1 again.  

What this repo contains:
- `main.py`: Main code for training model.
- `vae_module.py`: A training program for VAE, running in main.py.
- `gmm_module.py`: A training program for GMM, running in main.py.
- `tool.py`: Various functions handled in the program.



# How to run
You can train the VAE+GMM model by running `main.py`.  
- `train_model()` can be made to train VAE+GMM.  
- `decode_from_gmm_param()` makes image reconstruction from parameters of posterior distribution estimated by GMM.
```python:main.py
def main():
    # training VAE+GMM model
    train_model(mutual_iteration=2, # The number of mutual learning
                dir_name=dir_name, 
                train_loader=train_loader, # Dataloader for training
                all_loader=all_loader) # Dataloader when inference latent variables for all data points
    
    # reconstruct image
    load_iteration = 1 # Which iteration of the mutual learning model to load
    decode_from_gmm_param(iteration=load_iteration, 
                          decode_k=1, # The cluster number of the Gaussian distribution to be used as input for decoder.
                          sample_num=16, # The number of samples for the random variable.
                          model_dir=dir_name)
```

You need to have pytorch >= v0.4.1 and cuda drivers installed.  
My environment is the following **Pytorch==1.5.1+cu101, CUDA==10.1**  


# Changes with and without mutual learning (for MNIST)  
## Latent space on VAE  
Left : without mutual learning・Right : with mutual learning  
Plot using TSNE  
<div>
	<img src='/image/z_tsne_0.png' width="380px"><img src='/image/z_tsne_1.png' width="380px">
</div>
Plot using PCA  
<div>
	<img src='/image/z_pca_0.png' width="380px"><img src='/image/z_pca_1.png' width="380px">
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
	<img src='/image/acc_0.png' height="280px"><img src='/image/acc_1.png' height="280px">
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
The implementation of GMM is based on 
[【Python】4.4.2：ガウス混合モデルにおける推論：ギブスサンプリング【緑ベイズ入門のノート】](https://www.anarchive-beta.com/entry/2020/11/28/210948)

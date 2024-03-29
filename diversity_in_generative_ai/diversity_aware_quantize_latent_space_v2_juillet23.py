# -*- coding: utf-8 -*-
"""diversity_aware_quantize_latent_space_v2_juillet23.ipynb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:00:20 2023

Does the following :

1/ uses as input the weights from a run of the CVAE from
tensorflow tutorial
2/ uses as input the latent space of the mnist as encoed through CVAE
3/ with different kernels quantizes the latent MNIST
4/ outputs the quantized points and the decoded images

Note: now the learning rate is set to keras/Adam defaults. Can change them
by uncommenting in the beginning in the lines
#adam_learning_rate=0.001
#run_id=run_id+"_learn_rate_"+str(adam_learning_rate)
and later uncoment line
##opt = tf.keras.optimizers.Adam(learning_rate=adam_learning_rate)

Note: this is not too fast in general can set learning_rate=0.01 or even 0.1
not 0.001 as in Keras default...

@author: turinici
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
#from scipy.special import gamma as euler_gamma_function#only used if latent dimension is not 2

#parameters
run_id="_run_33"
distance_kernel='Huber-energy'
#distance_kernel='Gaussian'
run_id=run_id+"_kernel_"+distance_kernel+"_"
nr_iter=10000
sigma_kernel=np.sqrt(np.pi)#default value for latent_dim=2
# for general dimension this is
# 2.0* euler_gamma_function(latent_dim/2+1/2)/euler_gamma_function(latent_dim/2)
sigma_kernel=100.0#
run_id=run_id+"_iter"+str(nr_iter)
if(distance_kernel=='Gaussian'):
    run_id=run_id+"_sigma_"+str(sigma_kernel)

adam_learning_rate=0.001#0.001#Keras Adam default=0.001
run_id=run_id+"_learn_rate_"+str(adam_learning_rate)


print("run_id=",run_id)
################ end of parameter set

#%%

with np.load('reparam.npz') as data:
    reparam = data['reparam']
#reparam_data=np.load('reparam.npz')
#reparam=reparam_data.files['reparam']

#%%
print('mean=',np.mean(reparam,axis=0),'\n variance=',np.cov(reparam.T),
      '\n corr=',np.corrcoef(reparam.T))

#%%
print('plotting the encoded MNIST')
fig=plt.figure("encoded")
#choose some indices
rnd_indices=np.random.choice(list(range(reparam.shape[0])),size=5000,replace=False)
plt.scatter(reparam[rnd_indices,0],reparam[rnd_indices,1],s=.5)
#fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
#fig.show()
fig.tight_layout()
fig.savefig('encoded_mnist_reparametrized'+run_id+'.pdf',bbox_inches='tight')


#%%
K=10#K=3*16#number of Dirac masses in the compressed distribution
N=2#ambient dimension
J=1000#how many samples are included: more the better, otherwise there are oscillations in the SGD/Adam convergence

quant=tf.Variable(np.random.rand(N,K),dtype=tf.float32)#this will be the set of quantization variables

def loss(quant,batch):
  #sum_dist_normal=tf.math.sqrt ...
  return None

#%%
print('define empirical sampling')

def empirical_sampling(tmpN=reparam.shape[1],tmpJ=J):
    ''' sample from the empirical, target distribution : size as Y
    Here the target is the 'reparam' database, dim =N=2
    '''
    assert tmpN==reparam.shape[1], "wrong dimension N"
    rnd_indices=np.random.choice(list(range(reparam.shape[0])),size=tmpJ,replace=False)

    return reparam[rnd_indices,:].T


#%%
"""#Empirical Radon Sobolev distance"""

order_kernel=1#this is the main situation
#order_kernel=.5#this is square root idea
def empirical_radon_sobolev_distance_sq(X,Y,local_alphas=None,betas=None):
    '''
    Parameters
    ----------
    X : 2D NxK matrix
        input data sample, each column a vector of dimension N, notation X_k
    Y : same as X for the second distribution
    alphas : 1D array of weights for X
    betas : same as alphas for Y

    Note: X,alphas are Tensor/Variable while Y/betas are numpy array

    Returns
    -------
    Radon-Sobolev distance
    '''
    N,K=X.shape
    Ny,J=Y.shape
    Ytensor = tf.convert_to_tensor(Y, dtype=tf.float32)

    if local_alphas is None:
      alphas = tf.convert_to_tensor(np.ones(K)/K,dtype=tf.float32)
    else:
      alphas=tf.nn.softmax(local_alphas)
    if betas is None:
        betas = np.ones(J)/J
    betastensor = tf.convert_to_tensor(betas, dtype=tf.float32)

    assert (tf.rank(X).numpy()==2) & (Y.ndim==2) & (tf.rank(alphas).numpy()==1) & (betas.ndim==1),"invalid input dimensions"
    Ka,=alphas.shape
    Jb,=betas.shape
    assert (N==Ny)& (K==Ka)&(J==Jb), 'invalid input dimensions'
    points=tf.concat([X,Ytensor],axis=1)
    gammas=tf.concat([alphas,-betastensor],axis=0)
    #one liner for Huber-energy kernel
    #h_distZZ_sq = tf.math.pow(1.0e-10+tf.math.reduce_sum(tf.square(tf.expand_dims(points,2)-tf.expand_dims(points,1)),axis=0),order_kernel/2)-1.0e-10**(order_kernel/2)

    #General formula ; step 1 :compute distance squared
    distZZ_squared = tf.math.reduce_sum(tf.square(tf.expand_dims(points,2)-tf.expand_dims(points,1)),axis=0)
    #step 2: apply the kernel
    if(distance_kernel=='Huber-energy'):
        h_distZZ_sq = tf.math.pow(1.0e-10+distZZ_squared,order_kernel/2)-1.0e-10**(order_kernel/2)#Huber-energy kernel
    #Gaussian kernel
    if(distance_kernel=='Gaussian'):
        h_distZZ_sq = 1.0-tf.math.exp(-distZZ_squared/(2*sigma_kernel**2))#Gaussian kernel


    return tf.squeeze(-0.5*tf.expand_dims(gammas,0)@h_distZZ_sq@tf.expand_dims(gammas,1))
#debug and tests
print(empirical_radon_sobolev_distance_sq(tf.ones([N,K]),tf.zeros([N,J])))
print(empirical_radon_sobolev_distance_sq( tf.convert_to_tensor( np.array([[-1,1],[-1,1]]), dtype=tf.float32),
                                          np.array([[-1,-1,1,1],[-1,1,-1,1]])))

#%%

print('Main optimization loop')
#opt = tf.keras.optimizers.Adam()#Keras default parameters : (learning_rate=0.001,beta_1=0.9, beta_2=0.999
#opt = tf.keras.optimizers.Adam(learning_rate=0.1,beta_1=0.9, beta_2=0.999)
opt = tf.keras.optimizers.Adam(learning_rate=adam_learning_rate)
#can also use : opt.learning_rate.assign(0.0123)
##opt = tf.keras.optimizers.Adam()

loss_quant = lambda: empirical_radon_sobolev_distance_sq(quant,empirical_sampling())

loss_val=[]
opt_val=[]

for sc in tqdm(range(nr_iter)):
  opt.minimize(loss_quant, [quant])
  opt_val.append(quant.numpy())
  loss_val.append(loss_quant())

print('quant=',quant.numpy(),' step_count=',sc)
fig=plt.figure("quantization cv",figsize=(6,6))
plt.subplot(1,1,1)
plt.semilogy(loss_val)
df = pd.Series(loss_val)
plt.semilogy(np.array(df.ewm(span=10).mean()))
plt.savefig('mnist_latent_quantized_convergence_run'+run_id+'.pdf')

fig=plt.figure("quantization pts",figsize=(6,6))
plt.scatter(reparam[rnd_indices,0],reparam[rnd_indices,1],s=.5)
plt.scatter(np.array(opt_val)[-1,0,:],np.array(opt_val)[-1,1,:])
fig.show()
plt.savefig('mnist_latent_quantized_points'+run_id+'.pdf')


#%%

latent_quantized=quant.numpy().copy()
latent_quantized_save1=np.array([[-2.1528223 ,  1.2807379 , -0.0455162 , -0.64898515, -0.02833279,
        -0.67135614,  1.6630421 , -0.07095671, -0.8651524 ,  0.6661229 ],
       [ 0.20239553, -0.9379625 ,  0.22415748,  0.6480924 , -1.1955233 ,
        -2.1348174 ,  1.214619  ,  1.6738025 , -0.11518613,  0.2858877 ]],
      dtype=np.float32)
np.savez('quantization_results'+run_id+'.npz',latent_quantized=latent_quantized)

#%%
nbfilters1=32
nbfilters2=64


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    # self.encoder = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(28*28, activation='relu'),
    #         tf.keras.layers.Dense(28*28, activation='relu'),
    #         tf.keras.layers.Dense(28*28, activation='relu'),
    #         tf.keras.layers.Dense(28*28, activation='relu'),
    #         tf.keras.layers.Dense(28*28, activation='relu'),
    #         # last, No activation
    #         tf.keras.layers.Dense(latent_dim + latent_dim),
    #     ]
    # )

    self.encoder = tf.keras.Sequential(
        [tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),tf.keras.layers.Flatten()] +
        [tf.keras.layers.Dense(28*28, activation='relu') for ii in range(5)]
         + [tf.keras.layers.Dense(latent_dim + latent_dim)])


    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(28*28, activation='relu'),
            tf.keras.layers.Dense(28*28, activation='relu'),
            tf.keras.layers.Dense(28*28, activation='relu'),
            tf.keras.layers.Dense(28*28, activation='relu'),
            tf.keras.layers.Dense(28*28),
            tf.keras.layers.Reshape((28,28,1)),
        ]


    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


#%%
latent_dim=2
print('Create a new model instance')
model = CVAE(latent_dim)
res=model.sample(np.random.randn(1,latent_dim))
#generate_and_save_images(model, 0, test_sample)
model.built=True
print('restore weights')
model.load_weights("./my_checkpoint")#attention: need to be in the directory where the file "my_checkpoint.index" is

#%%
#plot nx x ny = 10 images from the generated distribution
nx=5
ny=2
digit_size=28
image_width = digit_size*nx
image_height = digit_size*ny

if(latent_dim==2):

    image3 = np.zeros((image_height, image_width))
    for ix in range(nx):
      for iy in range(ny):
        x_decoded = model.sample(latent_quantized[:,ix*ny+iy][None,:])
        digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
        image3[iy * digit_size: (iy + 1) * digit_size,
              ix * digit_size: (ix + 1) * digit_size] = digit.numpy()

    fig=plt.figure(figsize=(10,4))
    plt.imshow(image3, cmap='Greys_r')
    plt.axis('Off')
    plt.tight_layout()
    fig.show()
    plt.savefig('mnist_latent_quantized_samples'+run_id+'.pdf')

#results :
# Huber-energy learning-rate 0.01, iterations 1000-2000
# Gaussian kernel
# kernel 1000, lr=0.01, iter=200 : not converged, images poor quality
# kernel=1000  lr=0.01, iter=1k : not ok yet, slighter better, proably never better
# kernel=1  lr=0.01, iter=1k : presque ok, not yet converged
# kernel=1  lr=0.1, iter=1k : oscillations ... almost ok
# kernel=np.sqrt(np.pi)  lr=0.1, iter=1k : oscillations ...
# kernel=np.sqrt(np.pi)  lr=0.01, iter=1k : small oscillations but not converged
# kernel=np.sqrt(np.pi)  lr=0.01, iter=4k : oscillations quality not so good
# kernel=np.sqrt(np.pi)  lr=0.001, iter=4k : not so big oscillations but not converged


# rq: if all is ok, the mean distance should correspond to mean distance between
# two standard normals ; for dimension 2 this is np.sqrt(np.pi)

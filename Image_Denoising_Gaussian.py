'''
    File name: Image_Denoising_Gibbs.py
    Description: Image denoising using MRF and simple Gibb's sampling to estimate the posterior mean.
    Author: Pritam Chanda
    Date created: 12/10/2015
    Date last modified: 12/08/2017
    Python Version: 2.7
'''


from __future__ import division
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from Ising import IsingModel #contains the code for Gibb's sampling from MRF.
import cv2

class Image_Denoising(IsingModel):
   
   def __init__(self,Y,alpha,beta,sigma_sqrd):
      height,width = Y.shape
      #X = np.random.random([height,width])
      #X = np.where(X>=0.5,1,-1) # Y = de-noised image initialized with a random matrix.
      X = Y.copy()
      super(Image_Denoising,self).__init__(X,alpha,beta)
      self.ss = sigma_sqrd
      self.Y = Y  # Y = Observed noisy image.
   
   def conditional_probability(self,i,j):
      #print('cond denoise')
      neighbors = self.get_neighbors(i,j)
      energy = -self.alpha + self.beta*np.sum( self.X[x,y] for (x,y) in neighbors ) 
      term1 = np.exp(energy - (0.5/self.ss)*((self.Y[i,j]-1)**2) )
      term2 = np.exp(-energy - (0.5/self.ss)*((self.Y[i,j]+1)**2) )
      Prob_Xij_one_given_neighbors = float(term1)/(term1+term2)
      Prob_Xij_minus_one_given_neighbors = 1.0 - Prob_Xij_one_given_neighbors #float(term2)/(term1+term2)
      return [Prob_Xij_minus_one_given_neighbors,Prob_Xij_one_given_neighbors]


#read grayscale image, convert to binary
im = cv2.imread('bayes.jpg')

im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #convert to grayscale
_,im_binary = cv2.threshold(im,127,1,cv2.THRESH_BINARY)
im_binary = (im_binary.astype(np.int) * 2) - 1

#create a noisy binary image
noise = np.random.random(im_binary.shape)
noise = where(noise<0.15,-1,1)
Y = np.array(im_binary*noise) #observed noisy binary image.

height,width = Y.shape
N = height*width

#imgplot = plt.imshow(Y, cmap = cm.Greys_r)
#plt.show()

alpha = 0.0
beta = 0.3
sigma_squared = 1.4
num_burnin = 100
num_samples = 20
N = height*width

IMD = Image_Denoising(Y,alpha,beta,sigma_squared)

print('Running %d burn-in iterations...'%(num_burnin))
#burn-in iterations.
for n_burn_in in range(num_burnin):
   print('burin_in = %d '%(n_burn_in*N))
   for i in range(0,height):
      for j in range(0,width):
         IMD.Gibbs_sampling([i,j])

print('Sampling %d samples using Gibbs...'%(N*num_samples))
#now generate samples and take average.
avg = np.zeros_like(Y)

for n_samp in range(num_samples):
  print('s = %d '%(n_samp*N))
  for i in range(0,height):
      for j in range(0,width):
          IMD.Gibbs_sampling([i,j])
          avg += IMD.X

avg = avg.astype(float)
avg = avg/(N*num_samples)
avg[avg >= 0] = 1
avg[avg < 0] = -1
avg = avg.astype(np.int)

plt.figure('noisy image')
imgplot = plt.imshow(Y, cmap = cm.Greys_r)

plt.figure('posterior mean after sampling')
imgplot = plt.imshow(avg, cmap = cm.Greys_r)

plt.show()


'''
    File name: Image_Denoising_SA.py
    Description: Image denoising using MRF and simple Gibb's sampling with annealing to estimate the posterior mean.
    Author: Pritam Chanda
    Date created: 12/10/2015
    Date last modified: 12/08/2017
    Python Version: 2.7
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import cv2

class Image_Denoising_SA:

  def __init__(self,Y,alpha,beta,eta):
    height,width = Y.shape
    self.height = height
    self.width = width
    self.alpha = alpha
    self.beta = beta
    self.eta = eta
    self.Y = Y
    #self.X = Y.copy()
    X = np.random.random([height,width])
    X = np.where(X>=0.5,1,-1) # Y = de-noised image initialized with a random matrix.
    self.X = X

  def get_neighbors(self,i,j):
    neighbors = []
    neighbors.extend(([i-1,j+1],[i,j+1],[i+1,j+1]))
    neighbors.extend(([i-1,j],[i+1,j]))
    neighbors.extend(([i-1,j-1],[i,j-1],[i+1,j-1]))
    #take care of the boundaries.
    for n in xrange(len(neighbors)):
      [a,b] = neighbors[n]
      a = 0 if a<0 else a
      b = 0 if b<0 else b
      a = self.height-1 if a>=self.height else a
      b = self.width-1 if b>=self.width else b
      neighbors[n] = [a,b]
    #remove all occurrences if [i,j] from neighbors
    neighbors[:] = [a for a in neighbors if a != [i,j]]
    return neighbors


  def conditional_probability(self,i,j,Temp):
      neighbors = self.get_neighbors(i,j)
      energy = (self.alpha - self.beta*np.sum( self.X[x,y] for (x,y) in neighbors ) - self.eta*self.Y[i,j])/Temp
      term1 = np.exp(-energy)
      term2 = np.exp(energy)
      Prob_Xij_one_given_neighbors = float(term1)/(term1+term2)
      Prob_Xij_minus_one_given_neighbors = 1.0 - Prob_Xij_one_given_neighbors #float(term2)/(term1+term2)
      return [Prob_Xij_minus_one_given_neighbors,Prob_Xij_one_given_neighbors]


  def Gibbs_sampling(self,z,Temp):
    # z = [i,j]
    # sample from P( X[z] | N(X[z]) )
    i,j = z
    p_minus_one,p_one = self.conditional_probability(i,j,Temp)
    self.X[i,j]=-1 if np.random.rand()<=p_minus_one else 1 

 
#Main
def main():
  #read grayscale image, convert to binary
  im = cv2.imread('bayes.jpg')
  im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #convert to grayscale
  _,im_binary = cv2.threshold(im,127,1,cv2.THRESH_BINARY)
  im_binary = (im_binary.astype(np.int) * 2) - 1

  #create a noisy binary image
  noise = np.random.random(im_binary.shape)
  noise = where(noise<0.1,-1,1)
  Y = np.array(im_binary*noise) #observed noisy binary image.

  height,width = Y.shape
  N = height*width

  #imgplot = plt.imshow(Y, cmap = cm.Greys_r)
  #plt.show()

  alpha = 0.0
  beta = 0.5
  eta = 0.5
  max_iter = 200
  num_samples = 20
  N = height*width
  C = 4.0

  IMD = Image_Denoising_SA(Y,alpha,beta,eta)

  print('Running %d iterations...'%(max_iter))
  temperature = 0.0
  for n_iter in range(1,max_iter):
    temperature = C/np.log(n_iter+1)
    print('iter = %d, temperature = %f '%(n_iter,temperature))
    for i in range(0,height):
       for j in range(0,width):
          IMD.Gibbs_sampling([i,j],temperature)
    #if n_iter % 100 == 0:
    #   plt.figure('denoised image')
    #   imgplot = plt.imshow(IMD.X, cmap = cm.Greys_r)
    #   plt.show()

  avg = np.zeros_like(Y)
  for n_samp in range(1,num_samples):
    print('samples = %d '%(n_samp*N))
    temperature = C/np.log(n_samp+1)
    for i in range(0,height):
       for j in range(0,width):
          IMD.Gibbs_sampling([i,j],temperature)
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



if __name__== '__main__':
   main()

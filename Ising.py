'''
    File name: Ising.py
    Author: Pritam Chanda
    Date created: 11/20/2015
    Date last modified: 12/08/2017
    Python Version: 2.7
'''

from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

verbose = False

class IsingModel:

  def __init__(self,X,alpha,beta):
    # X = height x width
    # Gibbs distribution parameters : alpha, beta
    height,width = X.shape
    self.height = height
    self.width = width
    self.alpha = alpha
    self.beta = beta
    self.X = X


  def show(self,name):
      plt.figure(name)
      plt.imshow(self.X,cmap=cm.Greys_r)
      plt.show()


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


  def unnormalized_probability(self):
    energy = self.alpha * np.sum(self.X)
    pairwise = 0
    for i in range(self.height):
      for j in range(self.width):
          neighbors = self.get_neighbors(i,j)
          for neighbor_index in neighbors:
              pairwise += self.X[i,j]*self.X[neighbor_index[0],neighbor_index[1]]
    pairwise /= 2 #as each pair is counted twice 
    energy += -self.beta * pairwise
    if verbose:
       print('energy=%f'%(energy))
    return np.exp(-energy)


  def conditional_probability(self,i,j):
    neighbors = self.get_neighbors(i,j)
    energy = self.alpha - self.beta*np.sum( self.X[x,y] for (x,y) in neighbors )
    term1 = np.exp(-energy)
    term2 = np.exp(energy)
    Prob_Xij_one_given_neighbors = float(term1)/(term1+term2)
    Prob_Xij_minus_one_given_neighbors = 1.0 - Prob_Xij_one_given_neighbors #float(term2)/(term1+term2)
    return [Prob_Xij_minus_one_given_neighbors,Prob_Xij_one_given_neighbors]


  def Gibbs_sampling(self,z):
    # z = [i,j]
    # sample from P( X[z] | N(X[z]) )
    i,j = z
    p_minus_one,p_one = self.conditional_probability(i,j)
    self.X[i,j]=-1 if np.random.rand()<=p_minus_one else 1 

 
  # convert a binary matrix to an integer.
  def encode_binary_image(self):
      pos=0
      num = 0
      for i in range(0,self.height):
         for j in range(0,self.width):
             x = (self.X[i,j]+1.0)/2.0
             num += (2**pos)*x
             pos += 1
      return int(num)
  

# return a list of all possible binary matrices of size height x width
# results will be returned in matrix Z : height x width x 2^(height*width)

def gen_all_possible_matrices(X,pair,Z):
   i,j = pair
   n,m = X.shape
   if i==n:
     Z[:,:,gen_all_possible_matrices.count] = X
     gen_all_possible_matrices.count += 1
     return

   i_new = i; j_new = j
   if j<m-1:
    j_new = j + 1
   else:
    j_new = 0
    i_new = i + 1

   X[i,j]=-1; gen_all_possible_matrices(X,[i_new,j_new],Z)
   X[i,j]=1;  gen_all_possible_matrices(X,[i_new,j_new],Z)




# generate all possible binary images(matrices) of size height x width and compute their probabilities.
def actual_distribution(height,width,alpha,beta):
    X = np.zeros([height,width])-1
    Z = np.zeros([height,width,2**(height*width)])
 
    print('Generating all possible binary images of size %d x %d'%(height,width))
    gen_all_possible_matrices(X,[0,0],Z)

    #compute the probabilities for each of the matrices.
    prob = []
    partition_function = 0
    for k in range(0,2**(height*width)):
        mat = Z[:,:,k]
        modl = IsingModel(mat,alpha,beta)             
        pr = modl.unnormalized_probability()
        num = modl.encode_binary_image()
        prob.append([num,pr])
        partition_function += pr
        if verbose: 
          print('num = %d pr = %f\n'%(num,pr))
          print(mat)
          print('--------------------')
    if verbose: print('partition_function = ',partition_function)
    prob = [ (n,pr/partition_function) for (n,pr) in prob]    
    prob.sort(key = lambda pair: pair[1]) #sort by image encoded number
    return prob
 


#compare actual distribution with sampled distribution.
def compare(height,width,alpha,beta,num_burnin=10000,num_samples=20000):

    print('Comparing probablities from actual image distribution with that from Gibbs sampling using %d x %d image...'%(height,width))
    N = height*width
    #prob_actual has the exact probabilities of each image
    prob_actual = actual_distribution(height,width,alpha,beta)

    #now we will draw samples from P(X) using Gibbs sampling.
    #start at a random binary matrix X.
    X = np.random.rand(height,width)
    X = np.where(X<=0.5,-1,1) 
    ISM = IsingModel(X,alpha,beta)
    
    print('Running %d burn-in iterations...'%(num_burnin))
    #burn-in iterations.
    for _ in range(num_burnin):
      for i in range(0,height):
         for j in range(0,width):
            ISM.Gibbs_sampling([i,j])
    
    print('Sampling %d samples using Gibbs...'%(N*num_samples))
    #now generate samples
    count = [0]*(2**N)
    for _ in range(num_samples):
      for i in range(0,height):
         for j in range(0,width):
            ISM.Gibbs_sampling([i,j])
            numb = ISM.encode_binary_image()
            count[numb] += 1

    total = sum(count)
    #print('total=',total)
    count = [ x/total for x in count ] 
    
    ymax = max(prob_actual,key=lambda pair:pair[1])[1]   

    plt.plot(range(0,2**N),[count[x] for x,_ in prob_actual],'ro',label='Gibbs_sampling')
    plt.plot(range(0,2**N),[y for _,y in prob_actual],'b-',linewidth=2.0,label='Actual distribution')
    plt.axis([-1, len(prob_actual), -0.001, ymax])
    plt.legend()
    plt.show()


# Main
gen_all_possible_matrices.count = 0

height = 3
width = 3
alpha = 0.01
beta = 0.05
compare(height,width,alpha,beta)


import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = np.dot(X[i],W)                            #1xc
    scores -= np.max(scores)
    exp_scores = np.exp(scores)                        #1xc
    sum_exp_scores = np.sum(exp_scores)                #1xc
    inv_sum_exp_scores = 1/sum_exp_scores              #1xc
    y_hat = exp_scores * inv_sum_exp_scores            #1xc
    k = y[i]                                
    y_input = np.zeros(W.shape[1])                     #1xc
    y_input[k] = 1
    dW += np.outer(X[i], (y_hat - y_input))            #(d,).(c,) = dxc
    loss += -np.log(y_hat[y[i]])
  loss = loss/ num_train
  dW = dW/num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #print 'score.shape = ', score[1]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n = np.arange(y.shape[0])
  num_train = y.shape[0]
  c = W.shape[1]
  
  scores = np.dot(X,W)                                 #NxC  
  scores -= np.max(scores, axis = 1,keepdims = True)   #Nx1
  exp_scores = np.exp(scores)                          #NxC
  corr_scores = np.reshape(exp_scores[n,y],(exp_scores.shape[0],-1))               #Nx1  dscores[n,y] = 1*dcorr_scores
                                                                                   #exp_corr_scores = np.exp(corr_scores)
  sum_exp_scores = np.sum(exp_scores, axis = 1, keepdims = True)+1e-8      #Nx1
  inv_sum_exp_scores = 1/sum_exp_scores                #N,1
  f = corr_scores * inv_sum_exp_scores                 #(Nx1)*(Nx1) = (Nx1)
  lf = np.log(f)                                       #Nx1
  nlf = -1*lf
  loss = np.sum(nlf)                                   #1
    
  dnlf = np.ones((num_train,1))                        #Nx1
  dlf = -1 * dnlf                                      #Nx1
  df = (1/f) * dlf                                           #(Nx1)
  dcorr_scores = inv_sum_exp_scores * df                #(N,1).(N,1) = (N,1)
  dinv_sum_exp_scores = corr_scores * df                #(Nx1).(Nx1) = (Nx1)
  dsum_exp_scores = (-1/(sum_exp_scores ** 2))* dinv_sum_exp_scores        ##(Nx1).(Nx1) = (Nx1)
  dexp_scores = np.outer(dsum_exp_scores,np.ones(c))                                   #Nxc 
  dcorr_scores = np.reshape(dcorr_scores, (num_train,))   
  dexp_scores[n,y] += dcorr_scores   
  
  #print dcorr_scores.shape
  #print dinv_sum_exp_scores.shape
  #dexp_scores += dcorr_scores
  
  dscores = np.exp(scores) * dexp_scores                   #NxC
  # dcorr_scores = np.exp(corr_scores) * dexp_corr_scores
  #dscores[n,y] += 1 * dcorr_scores
  #print dscores.shape
  dW = np.dot(X.T , dscores)                                  #
  dX = np.dot(dscores , W.T)
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


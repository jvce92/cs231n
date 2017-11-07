import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0
  dW = np.zeros(W.shape)
  for i in np.arange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in np.arange(num_classes):
      if j == y[i]:
          continue
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        count += 1
        dW[:, j] += X[i]
        loss += margin
    dW[:, y[i]] -= count * X[i]
      
  loss = (loss/num_train + (reg/2) * np.sum(W*W))
  dW = dW/num_train + reg * W
  
  return loss, dW

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0
  num_train = X.shape[0]
  dW = np.zeros(W.shape)
  scores = X.dot(W)
  correct_scores = scores[np.arange(scores.shape[0]), y]
  margins = np.maximum(0, scores - correct_scores[:, np.newaxis] + 1)
  margins[np.arange(margins.shape[0]), y] = 0
  loss = (np.sum(margins)/num_train) + (1/2) * reg * np.sum(W*W)
  mask = np.zeros_like(margins)
  mask[margins > 0] = 1
  count = np.sum(mask, axis=1)
  mask[np.arange(mask.shape[0]), y] = -count
  dW = X.T.dot(mask)
  dW = dW/num_train + reg * W
  return loss, dW

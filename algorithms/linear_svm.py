import numpy as np
from random import shuffle

###############################################################################
#                                                                             #
# Python implementation of a multi-class support vector machine classifier.   #
# Code developed as part of solution to CS231n Convolutional Neural Networks  #
# for visual recognition.                                                     #
# See http://cs231n.github.io/assignments2016/assignment1/                    #
# for detail instructions.                                                    #
###############################################################################

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are K classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, K) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
  dW = (dW) / num_train
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  
  delta = 1.0
  num_train = X.shape[0]
  
  # Matrix of scores - each row is a data example of scores across classes.
  scores = (X.dot(W)).T
  correct_class_scores = scores[y, np.arange(num_train)]
  print(scores.shape)
  print(correct_class_scores.shape)  
  si = scores - correct_class_scores + delta
  s = si.clip(min=0).sum(axis=0)
  s -= delta
  loss = np.mean(s) + 0.5* reg * np.sum(W*W)
  
  hinged = np.zeros(scores.shape)
  
  hinged[si > 0] = 1.0
  
  num_positive = np.sum(si > 0, axis=0)
  hinged[y, np.arange(num_train)] -= num_positive
  
  dW = hinged.dot(X) / num_train + reg * W.T
  
  return loss, dW.T
import numpy as np
from random import shuffle

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
  #dw = 0.0
  #cnt=0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    #print(scores.shape)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        #print(X[i])
        #cnt +=1
  dW = (dW) / num_train
  #print(dW.shape)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #print(dW.shape)  
  #print(dW[3000, 9])

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  #dW = np.zeros(W.shape) # initialize the gradient as zero
  
  delta = 1.0
  #num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # Matrix of scores - each row is a data example of scores across classes.
  #scores = X.dot(W)
  scores = (X.dot(W)).T
  correct_class_scores = scores[y, np.arange(num_train)]
  #print(si.shape)
  print(scores.shape)
  print(correct_class_scores.shape)  
  si = scores - correct_class_scores + delta
  s = si.clip(min=0).sum(axis=0)
  s -= delta
  loss = np.mean(s) + 0.5* reg * np.sum(W*W)
  
  indicators = np.zeros(scores.shape)
  
  indicators[si > 0] = 1.0
  
  num_positive = np.sum(si > 0, axis=0)
  indicators[y, np.arange(num_train)] -= num_positive
  
  dW = indicators.dot(X) / num_train + reg * W.T
  #print(si)
  #print(num_train)
  #print(scores.shape)
  #print(y)
  #print(correct_class_scores)
  #correct_class_scores = scores[y, np.arange(num_classes)]
  #print(correct_class_scores.shape)
  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW.T

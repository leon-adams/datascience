# notebooks
A collection of data analysis and machine learning notebooks.

## linear-classifier
This notebooks uses Numpy arrays and graphlab SFrame as the base datastructures to implement the logistic 
regression algorithm. We directly implement the logistic link function, compute the log-likelihood function,
implement gradient descent and compute the classification accuracy. Dataset used is a subset of Amazon product
reviews accessed from module 3 of the Coursera Machine learning specilization [1]. 

## linear-classification-regularization
This notebook extends the work of the linear classifier above. Here, emphasis is placed on the use of 
regularization to offset the effects of overfitting in the context of a logistic classifier. The log-likelihood 
function is extended to include the L2 penalty.

## binary-decision-tree
This notebook implements a binary decision tree classifier. Again, starting with Numpy arrays and graphlab
SFrame datastructues we write code:
* Build binary decision tree
* Make predictions using the constructed decision tree
* Evaluate the accuracy of the constructed decision tree

Dataset for this module was the lending club dataset provided within module 3 of the Coursera Machine Learning
Specialization [1].

## decision-tree-boosting
This module extends the work of the binary-decision-tree classifier above. Here, emphasis is placed on improving
model accuarcy through the use of boosting. We modify the binary-decision tree to inculde the use of weighted
data points. Adaboost is directly implemented and we evaluate the effect of boosting on model performance.


[1] Coursera Machine Learning Specialization. Available at: https://www.coursera.org/specializations/machine-learning

[2] CS231n: Convolutional Neural Networks for Visual Recognition. Available at: http://cs231n.stanford.edu/

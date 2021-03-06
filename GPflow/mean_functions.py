import tensorflow as tf
import numpy as np
from param import Param, Parameterized
import transforms

class MeanFunction(Parameterized):
    """
    The base mean function class.

    To implement a mean funcion, write the __call__ method. This takes a 
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X. 

    MeanFunction classes can have parameters, see the Linear class for an example.
    """
    def __call__(self, X):
        raise NotImplementedError, "Implement the __call__ method for this mean function"
    
class Zero(MeanFunction):
    def __call__(self, X):
        return tf.zeros(tf.pack([tf.shape(X)[0], 1]), dtype='float64')


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=np.ones((1,1)), b=np.zeros(1)):
        """
        A is a matrix which maps each element of X to Y, b is an additive constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q. 
        """
        MeanFunction.__init__(self)
        self.A = Param(np.atleast_2d(A))
        self.b = Param(b)
    def __call__(self, X):
        return tf.matmul(X, self.A) + self.b


class Constant(MeanFunction):
    """
    y_i = c
    """
    def __init__(self, c=np.zeros(1)):
        MeanFunction.__init__(self)
        self.c = Param(c)
    def __call__(self, X):
        return tf.tile(tf.reshape(self.c, (1,-1)), tf.pack([tf.shape(X)[0], 1]))


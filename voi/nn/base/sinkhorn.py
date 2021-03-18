import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# This uses tf.py_function to call scipy.optimize.linear_sum_assignment,
# which is slow in multi-gpu setting (as tf only allows one py_function to run
# in the address space of the program
def hungarian(x):
    if x.ndim == 2:
        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        sol[i, :] = linear_sum_assignment(x[i, :])[1].astype(np.int32)
    return sol

# Reimplementation of Hungarian algorithm in pure tensorflow ops,
# which is very slow when the matrix is large
# def findtwoinrow(mask, row):
#     mrow = tf.gather(mask, row, axis=0)
#     return tf.cast(tf.where(mrow == 2), dtype=tf.int32)

# def findoneinrow(mask, row):
#     mrow = tf.gather(mask, row, axis=0)
#     return tf.cast(tf.where(mrow == 1), dtype=tf.int32)

# def findoneincol(mask, col):
#     mcol = tf.gather(mask, col, axis=1)
#     return tf.cast(tf.where(mcol == 1), dtype=tf.int32)

# def make_onebytwo(a, b):
#     return tf.concat([a[tf.newaxis, tf.newaxis], 
#                       b[tf.newaxis, tf.newaxis]], axis=1)

# def findzero(cost, rowscovered, colscovered):
#     rowscovered = tf.cast(rowscovered, tf.float32)[:, tf.newaxis]
#     colscovered = tf.cast(colscovered, tf.float32)[tf.newaxis, :]
#     cost = cost + rowscovered * 10000 + colscovered * 10000
#     return tf.cast(tf.where(cost < 1e-20), dtype=tf.int32)

# def update_label(cost, rowscovered, colscovered):
#     rowscovered = tf.cast(rowscovered, tf.float32)[:, tf.newaxis]
#     colscovered = tf.cast(colscovered, tf.float32)[tf.newaxis, :]
#     tmpcost = cost + rowscovered * 10000000 + colscovered * 10000000
#     mincost = tf.reduce_min(tmpcost)
#     cost = cost + rowscovered * mincost - (1-colscovered) * mincost
#     return cost
    
# def hungarian(costs):
#     #s0, s1, _ = costs.get_shape().as_list()
#     s0 = tf.shape(costs)[0]
#     s1 = tf.shape(costs)[1]
#     costs = costs - tf.reduce_min(costs, axis=(-2, -1), keepdims=True) # turn x to non-negative
#     costs = costs - tf.reduce_min(costs, axis=-1, keepdims=True)
    
#     rowscovered = tf.zeros([s0, s1], dtype=tf.bool)
#     colscovered = tf.zeros([s0, s1], dtype=tf.bool)
#     masks = tf.zeros([s0, s1, s1], dtype=tf.int32)
#     i = 0#tf.zeros([], dtype=tf.int32)
#     j = 0#tf.zeros([], dtype=tf.int32)
#     while tf.less(i, s0):
#         while tf.less(j, s1):
#             idx = tf.where(tf.math.logical_and(
#                                 tf.math.logical_and(tf.cast(costs[:, i, j] < 1e-20, tf.bool), 
#                                                     tf.logical_not(rowscovered[:, i])),
#                                 tf.logical_not(colscovered[:, j]))) # 2-dim
#             idx = tf.cast(idx, tf.int32)
#             tmp = tf.ones([tf.shape(idx)[0], 1], dtype=tf.int32)
#             tmpi = tmp * i
#             tmpj = tmp * j
#             masks = tf.tensor_scatter_nd_update(masks, 
#                                                 tf.concat([idx, tmpi, tmpj], axis=1),
#                                                 tf.ones([tf.shape(idx)[0]], dtype=tf.int32))
#             rowscovered = tf.tensor_scatter_nd_update(rowscovered, 
#                                                 tf.concat([idx, tmpi], axis=1),
#                                                 tf.ones([tf.shape(idx)[0]], dtype=tf.bool))          
#             colscovered = tf.tensor_scatter_nd_update(colscovered, 
#                                                 tf.concat([idx, tmpj], axis=1),
#                                                 tf.ones([tf.shape(idx)[0]], dtype=tf.bool))
#             j += 1
#         i += 1
    
#     mask_arr = tf.zeros([1, s1, s1], dtype=tf.int32)
#     b = 0 #tf.zeros([], dtype=tf.int32)
#     while tf.less(b, s0):
#         tf.autograph.experimental.set_loop_options(
#             shape_invariants=[(mask_arr, tf.TensorShape([None, None, None]))])        
#         cost = costs[b]
#         rowscovered = tf.zeros([s1], dtype=tf.bool)
#         colscovered = tf.zeros([s1], dtype=tf.bool)        
#         mask = masks[b, :, :]
#         step = tf.ones([], dtype=tf.int32)
#         pathstartrow = tf.zeros([], dtype=tf.int32)
#         pathstartcol = tf.zeros([], dtype=tf.int32)
#         while tf.not_equal(step, -1):
#             if tf.equal(step, 1):
#                 cmasksum = tf.reduce_sum(mask, axis=0)
#                 cmasksum = tf.math.minimum(cmasksum, 1)
#                 colscovered = tf.cast(cmasksum, tf.bool)
#                 cmasksum = tf.reduce_sum(cmasksum)
#                 if cmasksum >= s1:
#                     step -= 2 # -1
#                 else:
#                     step += 1 # 2
#             elif tf.equal(step, 2):
#                 breakstep = tf.zeros([], dtype=tf.int32)
#                 while tf.not_equal(breakstep, 1):
#                     nonzeros = findzero(cost, rowscovered, colscovered)
#                     if tf.shape(nonzeros)[0] == 0:
#                         breakstep += 1
#                         step += 2 # 4
#                     else:
#                         locrow = nonzeros[0,0]
#                         loccol = nonzeros[0,1]
#                         mask = tf.tensor_scatter_nd_update(mask, 
#                                                            nonzeros[0][tf.newaxis, :], 
#                                                            tf.constant([2], dtype=tf.int32))
#                         next_cols = findoneinrow(mask, locrow)
#                         if tf.shape(next_cols)[0] == 0:
#                             pathstartrow = locrow
#                             pathstartcol = loccol
#                             breakstep += 1
#                             step += 1 # 3
#                         else:
#                             next_col = next_cols[0,0]
#                             rowscovered = tf.tensor_scatter_nd_update(rowscovered, 
#                                                            locrow[tf.newaxis, tf.newaxis], 
#                                                            tf.constant([True]))
#                             colscovered = tf.tensor_scatter_nd_update(colscovered, 
#                                                            next_col[tf.newaxis, tf.newaxis], 
#                                                            tf.constant([False]))                            
#             elif tf.equal(step, 3):
#                 twos = make_onebytwo(pathstartrow, pathstartcol)
#                 ones = make_onebytwo(pathstartrow, pathstartcol) # first row is dummy
#                 breakstep = tf.zeros([], dtype=tf.int32)
#                 while tf.not_equal(breakstep, 1):
#                     tf.autograph.experimental.set_loop_options(
#                         shape_invariants=[(twos, tf.TensorShape([None, 2])),
#                                           (ones, tf.TensorShape([None, 2]))])
#                     tmp_pathstartrow = findoneincol(mask, pathstartcol)
#                     if tf.shape(tmp_pathstartrow)[0] == 0:
#                         breakstep += 1
#                     else:
#                         pathstartrow = tmp_pathstartrow[0, 0]
#                         tmp = make_onebytwo(pathstartrow, pathstartcol)                        
#                         ones = tf.concat([ones, tmp], axis=0)
#                         tmp_pathstartcol = findtwoinrow(mask, pathstartrow)
#                         pathstartcol = tmp_pathstartcol[0, 0]
#                         tmp = make_onebytwo(pathstartrow, pathstartcol)
#                         twos = tf.concat([twos, tmp], axis=0)
#                 if tf.shape(ones)[0] > 1:
#                     ones = ones[1:]
#                     mask = tf.tensor_scatter_nd_update(mask,
#                                                        ones,
#                                                        tf.zeros([tf.shape(ones)[0]], dtype=tf.int32))
#                 mask = tf.tensor_scatter_nd_update(mask,
#                                                    twos,
#                                                    tf.ones([tf.shape(twos)[0]], dtype=tf.int32))                
#                 rowscovered = tf.zeros([s1], dtype=tf.bool)
#                 colscovered = tf.zeros([s1], dtype=tf.bool)       
#                 mask = mask - tf.cast(tf.equal(mask, 2), tf.int32) * 2
#                 step -= 2 # 1
#             elif tf.equal(step, 4):
#                 cost = update_label(cost, rowscovered, colscovered)
#                 step -= 2 # 2
#         mask_arr = tf.concat([mask_arr, mask[tf.newaxis, :, :]], axis=0)
#         b += 1
            
#     masks = mask_arr[1:]
#     return tf.stop_gradient(tf.cast(masks, tf.int32))
    
def matching(matrix_batch):
    """Solves a matching problem for a batch of matrices.
    Modified from 
    https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
    
    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem
    Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
      shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
      batch_size = 1.
    Returns:
    listperms, a 3D integer tensor of permutations with shape [batch_size, N, N]
      so that listperms[n, :, :] is the permutation matrix P of size N*N that solves the
      problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """

    listperms = tf.py_function(func=hungarian, inp=[matrix_batch], Tout=tf.int32) # 2D
    listperms.set_shape(tf.TensorShape([None, None]))
    #listperms = tf.one_hot(listperms, tf.shape(listperms)[1], dtype=tf.int32) # 3D
    return listperms
    #return hungarian(-matrix_batch)

def sinkhorn_loop_fn(x,
                     step,
                     iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix"""

    x = tf.math.log_softmax(x, axis=-2)
    x = tf.math.log_softmax(x, axis=-1)
    return x, step + 1, iterations


def sinkhorn_cond_fn(x,
                     step,
                     iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    condition: tf.Tensor
        a boolean that determines if the loop that applies
        the Sinkhorn Operator should exit"""

    return tf.less(step, iterations)


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=None, dtype=tf.int32)])
def sinkhorn(x,
             iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the sinkhorn operator"""

    args = [x, tf.constant(0, dtype=tf.int32), iterations]
    return tf.while_loop(
        sinkhorn_cond_fn, sinkhorn_loop_fn, args)[0]


class Sinkhorn(tf.keras.layers.Layer):

    def __init__(self,
                 iterations=20):
        """Calculate the result of applying the Sinkhorn Operator
        to a permutation matrix in log space

        Arguments:

        iterations: tf.Tensor
            the total number of iterations of the Sinkhorn operator
            to apply to the data matrix"""
        super(Sinkhorn, self).__init__()

        self.iterations = iterations

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a pointer network that generates
        permutation matrices in log space

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            a permutation matrix in log space that has the same shape
            as the transformer attention weights"""

        # apply the sinkhorn operator
        return tf.exp(sinkhorn(inputs, tf.constant(self.iterations)))

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(iterations=self.iterations)

        base_config = super(Sinkhorn, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

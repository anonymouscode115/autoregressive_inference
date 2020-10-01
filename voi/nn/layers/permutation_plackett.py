from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.input import AttentionInput
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import linear_sum_assignment
import numpy as np

#@tf.function
def deterministic_NeuralSort(logs, tau, mask):
    """s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar."""
    s = tf.math.exp(logs) * mask
    tf.print("s[0]", tf.squeeze(s[0]), summarize=-1)
    n = tf.shape(s)[1]
#     even_n = False
#     if n % 2 == 0:
#         even_n = True
#         n += 1
#         s = tf.concat([s, tf.zeros([tf.shape(s)[0], 1, 1])], axis=1)
#         mask = tf.concat([mask, tf.zeros([tf.shape(s)[0], 1, 1])], axis=1)
    one = tf.ones((n, 1), dtype = tf.float32)
    A_s = tf.math.abs(s - tf.transpose(s, [0, 2, 1]))
    B = tf.matmul(A_s, tf.matmul(one, tf.transpose(one)))
    scaling = tf.cast(n + 1 - 2*(tf.range(n) + 1), dtype = tf.float32)
    C = tf.matmul(s, tf.expand_dims(scaling, 0))
    P_max = tf.transpose(C-B, perm=[0, 2, 1])
    tf.print("P_max maximum", tf.reduce_max(tf.math.abs(P_max)))
#     P_hat = tf.nn.softmax(P_max / tau, -1)
    P_hat = P_max
    P_hat = P_hat - (1 - tf.transpose(mask, (0,2,1))) * 1000000000.0
#     if even_n:
#         P_hat = P_hat[:, :-2, :] 
#     else:
#         P_hat = P_hat[:, :-1, :]
    #P_hat = P_hat[:, :-1, :] # 0 is always masked out, which is always placed in the last
    #tf.print("phat", P_hat[0], summarize=-1)
    return P_hat     

#@tf.function
def custom_gather(a, b):
    """ Permutes the vectors in the last dimension of a according to
        the corresponding vectors in the last dimension of b
        e.g. a=[[1,4,7,6],[4,3,6,8]], b=[[0,3,1,2],[2,1,3,0]]
        Returns [[1,6,4,7],[6,3,8,4]]
    """
    original_shape = tf.shape(a)
    lastdim = tf.shape(a)[-1]
    a = tf.reshape(a, (-1, lastdim))
    b = tf.reshape(b, (-1, lastdim))
    idx = tf.range(tf.shape(a)[0])[:, tf.newaxis]
    idx = tf.tile(idx, [1, tf.shape(a)[1]])
    idx = tf.concat([idx[..., tf.newaxis], b[..., tf.newaxis]], axis=-1)
    result = tf.gather_nd(a, idx)
    result = tf.reshape(result, original_shape)
    return result        

def hungarian_shuffle(x, mask):
    if x.ndim == 2:
        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        tmp = sol[i, :]
        criterion = np.logical_and(tmp>0, tmp<=mask[i])
        sol[i, :] = np.concatenate([tmp[criterion],
                                    tmp[np.logical_not(criterion)]])
    return sol


#@tf.function
def matching2d(matrix_batch, mask):
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
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
      so that listperms[n, :] is the permutation matrix P of size N*N that solves the
      problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """
    mask = tf.reduce_sum(mask, axis=1)
    listperms = tf.py_function(func=hungarian_shuffle, inp=[matrix_batch, mask], Tout=tf.int32) # 2D
    listperms.set_shape(tf.TensorShape([None, None]))
    return listperms

def find_permu(matrix_batch):
    sol = np.zeros((matrix_batch.shape[0], matrix_batch.shape[1]), dtype=np.int32)
    flag = np.zeros_like(sol)
    rangeb = np.arange(matrix_batch.shape[0])
    for j in range(matrix_batch.shape[1]):
        tmp = matrix_batch[:,j,:]
        tmp = tmp - flag * 1000000000.0
        idx = np.argmax(tmp, axis=1)
        sol[:, j] = idx
        flag[rangeb, idx] = 1
    return sol
    
class PermutationPlackettLayer(Layer):

    def __init__(self,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 temperature=1.0,
                 use_gumbel_noise=True,
                 **kwargs):
        """Creates a Transformer permutation layer by applying a multi
        head sequence to matrix layer; and then create hard permutation
        through Plackett-Luce distribution
        Arguments:
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        temperature: float
            a positive number to divide the permutation logits by prior
        use_gumbel_noise: bool, UNUSED
            whether to apply gumbel noise to the output of PermutationLayer"""
        
        super(PermutationPlackettLayer, self).__init__()

        # the core attention and processing variables
        self.block0 = Block(hidden_size // 2,
                            1,
                            **kwargs)
        
        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.temperature = temperature
        self.use_gumbel_noise = use_gumbel_noise
        self.kwargs = kwargs   
    
    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of TransformerInput
        Arguments:
        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks
        Returns:
        permutation: TransformerInput"""

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        [queries, values, queries_mask, values_mask,
         pretrain_done, action_refinement, _, _, _, _, _, _, _,
         object_detections, object_features, object_boxes] = inputs   
        
        shape = tf.shape(queries)
        # log(s)
        activations = tf.maximum(tf.math.softplus(self.block0(queries, **kwargs)), 1e-5)
        activations = tf.math.log(activations)
        # prevent two activations being identical
        noise = tf.random.uniform(shape=tf.shape(activations), maxval=1e-5)
        activations += noise
        
        activations = tf.repeat(activations, action_refinement, axis=0) # (batch, len, 1)
        sqz_activations = tf.squeeze(activations, axis=2)
        
        queries_mask = tf.repeat(tf.expand_dims(queries_mask, 1), action_refinement, axis=0)
        valid_activation_mask = tf.cast(queries_mask, tf.float32) # (batch, 1, len)
        n_valid = tf.reduce_sum(valid_activation_mask, axis=-1)
        onedim_va_mask = tf.transpose(valid_activation_mask, [0,2,1])
        sqz_onedim_va_mask = tf.squeeze(onedim_va_mask, axis=2)
        masked_activations = tf.where(onedim_va_mask > 0, activations,
                               tf.ones_like(activations) * (-1000000.0))
                                                   
        twodim_va_mask = tf.matmul(valid_activation_mask, valid_activation_mask,
                                    transpose_a = True) # (batch, len, len)
        g = tfp.distributions.Gumbel(
                loc=tf.zeros_like(activations), 
                scale=tf.ones_like(activations))        
        perturb_acti = masked_activations + g.sample()
        perturb_acti = deterministic_NeuralSort(perturb_acti, self.temperature, onedim_va_mask)
        tf.print("perturb_acti[0]", perturb_acti[0], summarize=-1)
        id_permu = tf.cast(tf.range(shape[1])[tf.newaxis, :], tf.int32)
        #chosen_idx = tf.cast(matching2d(perturb_acti, sqz_onedim_va_mask), tf.int32)[:, :-1]
        chosen_idx = tf.py_function(func=find_permu, inp=[perturb_acti], Tout=tf.int32) # 2D
        chosen_idx.set_shape(tf.TensorShape([None, None]))   
        chosen_idx = chosen_idx[:, :-1]
        chosen_idx = tf.concat([tf.zeros([tf.shape(chosen_idx)[0], 1], dtype=tf.int32), chosen_idx],
                                axis=-1)
#         chosen_idx = tf.cast(tf.math.argmax(perturb_acti, axis=-1), tf.int32)[:, :-1]
#         chosen_idx = tf.concat([tf.zeros([tf.shape(chosen_idx)[0], 1], dtype=tf.int32), chosen_idx],
#                                 axis=-1)    
        onedim_sample_permu = tf.where(sqz_onedim_va_mask > 0, chosen_idx, id_permu)
        tf.print("onedim_sample_permu[:3]", onedim_sample_permu[:3], summarize=-1)
        tf.print("onedim_sample_permu", tf.reduce_sum(onedim_sample_permu, axis=-1), summarize=-1)
#         for i in range(tf.shape(onedim_sample_permu)[0]):
#             if tf.reduce_sum(onedim_sample_permu, axis=-1)[i] != 231:
#                 tf.print("nan activations", sqz_activations[i], summarize=-1)
#                 tf.print("nan perturb acti", perturb_acti[i], summarize=-1)
#                 tf.print("nan chosen idx", chosen_idx[i], summarize=-1)
#                 tf.print("nan mask", sqz_onedim_va_mask[i], summarize=-1)
#                 tf.print("nan matching", matching2d(perturb_acti)[i], summarize=-1)
                
        sample_permu = tf.one_hot(onedim_sample_permu, depth=shape[1], axis=-1)
        
#         tf.print("sample permu [:3]", sample_permu[:3], summarize=-1)
#         for idx in range(3):
#             locs = tf.where(sample_permu[idx] == 1.0)
#             d2 = tf.shape(locs)[1]
#             locs = tf.reshape(locs, [locs[-1,0]+1, d2])
#             tf.print("Sampled 3 permutations:",
#                      locs[:, -1], "\n", summarize=-1)        
            
        exp_actis = custom_gather(tf.squeeze(masked_activations, 2), onedim_sample_permu)
        exp_actis = tf.math.exp(exp_actis)
        reverse_cumsum_exp_actis = tf.math.cumsum(exp_actis[:, ::-1], axis=-1)[:, ::-1]
        eps = 1e-20
        log_nominator = tf.math.log(exp_actis + eps) - tf.math.log(reverse_cumsum_exp_actis + eps)
        log_nominator = log_nominator * sqz_onedim_va_mask
        tf.print("exp actis", exp_actis[0], summarize=-1)
        tf.print("reverse cumsum exp actis", reverse_cumsum_exp_actis[0], summarize=-1)
        tf.print("log_nominator[0]", log_nominator[0], summarize=-1)
        log_nominator = tf.reduce_sum(log_nominator, axis=-1, keepdims=True)
        tf.print("log_nominator", tf.squeeze(log_nominator), summarize=-1)  
        log_normalize_const = tf.zeros_like(log_nominator)
        
        # calculate kl divergence KL(X+eps || eps), where eps ~ gumbel noise
        kl_term1 = n_valid * (tf.math.log(self.temperature) - 1.0 \
                   + np.euler_gamma * (1.0 / self.temperature - 1.0))
        s1 = 1.0 / self.temperature \
            * tf.reshape(tf.reduce_sum(sqz_activations * sqz_onedim_va_mask, axis=-1), (-1,1))
        # numerical stability
        s2 = tf.reshape(tf.reduce_sum(tf.math.exp(-1.0 / self.temperature * tf.math.maximum(sqz_activations * sqz_onedim_va_mask, -20.0 * self.temperature)), \
                 axis=-1), (-1,1)) - (tf.cast(shape[1], tf.float32) - n_valid)
        kl = kl_term1 + s1 + s2 * tf.math.exp(tf.math.lgamma(1.0 + 1.0 / self.temperature))
        
        tf.print("pretrain_done", pretrain_done)
        tf.print("kl, s1, s2", tf.squeeze(kl), tf.squeeze(s1), tf.squeeze(s2), summarize=-1)         
        
        return [sample_permu, tf.squeeze(masked_activations, 2), kl, 
                log_nominator, log_normalize_const]

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process
        Returns:
        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      heads=self.heads,
                      queries_dropout=self.queries_dropout,
                      keys_dropout=self.keys_dropout,
                      temperature=self.temperature,
                      ** self.kwargs)

        base_config = super(PermutationPlackettLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
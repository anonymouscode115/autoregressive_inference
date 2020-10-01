from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.sequence_to_mat_sinkhorn import SequenceToMatSinkhorn
from voi.nn.base.sinkhorn import Sinkhorn
from voi.nn.base.sinkhorn import matching
from voi.nn.base.bethe import Bethe
from voi.nn.input import AttentionInput
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class PermutationSinkhornLayer(Layer):

    def __init__(self,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 iterations=200,
                 temperature=1.,
                 use_gumbel_noise=True,
                 **kwargs):
        """Creates a Transformer permutation layer by applying a multi
        head sequence to matrix layer; and then applying sinkhorn
        normalization to the activations
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
        iterations: tf.Tensor
            the total number of iterations of the Sinkhorn operator
            to apply to the data matrix
        temperature: float
            a positive number to divide the permutation logits by prior
            to applying sinkhorn normaliozation
        use_gumbel_noise: bool, UNUSED
            whether to apply gumbel noise to the output of PermutationLayer"""
        
        super(PermutationSinkhornLayer, self).__init__()

        # the core attention and processing variables
        self.sinkhorn = Sinkhorn(iterations=iterations)
        self.bethe = Bethe(iterations=30)
        self.sequence_to_mat = SequenceToMatSinkhorn(
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout)
        self.block0 = Block(hidden_size // 2,
                            hidden_size * 2,
                            **kwargs)
        
        # this tracks the batch of activation matrices before 
        # applying Gumbel noise and doing Sinkhorn 
        # operation, in order to calculate the prob of a 
        # permutation matrix using Gumbel-Matching potential
        self.activations = None
        # this tracks the kl divergence term between 2 distributions
        # KL((X+e)/self.temperature || e/temp_prior),
        # where e is the gumbel noise
        self.kl = 0.0
        
        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.iterations = iterations
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
        permutation: TransformerInput
            the result of applying a sequence to matrix layer and
            sinkhorn normalization; a doubly stochastic matrix
            with shape [batch, seq_length, seq_length]"""

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        [queries, values, queries_mask, values_mask,
         pretrain_done, action_refinement, _, _, _, _, _, _, _,
         object_detections, object_features, object_boxes] = inputs   
        
        shape = tf.shape(queries)
        hidden_dim = self.hidden_size // self.heads

        # pass the input through a feed forward processing block and
        # separate heads from channels
        activations = self.block0(queries, **kwargs)
        activations = tf.transpose(tf.reshape(activations, [
            shape[0], shape[1], self.heads, hidden_dim * 2]), [0, 2, 1, 3])

        # convert the inputs into the standard data class format expected
        # by the attention class
        queries_mask = tf.expand_dims(queries_mask, 1)
        attention_input = AttentionInput(
            queries=activations[..., :hidden_dim],
            keys=activations[..., hidden_dim:],
            queries_mask=queries_mask,
            values_mask=queries_mask)

        # pass the input through an attention processing block and
        # take the sum over the parallel attention heads
        activations = self.sequence_to_mat(attention_input, **kwargs)
        activations = tf.reduce_sum(activations, axis=1)
        
        # pass the outputs of the attention through a normalization layer
        # that performs sinkhorn normalization
        
        #print("debug activation:", self.activations[0])
#         n_valid = tf.reduce_sum(tf.cast(activations[:,1,:] > -1000000, tf.float32), \
#                                 axis=-1, keepdims=True)
#         nsquare_valid = n_valid ** 2
        valid_activation_mask = tf.cast(queries_mask, tf.float32)
        n_valid = tf.reduce_sum(valid_activation_mask, axis=-1)
        nsquare_valid = n_valid ** 2
        valid_activation_mask = tf.matmul(valid_activation_mask, valid_activation_mask,
                                    transpose_a = True)
        invalid_diags = tf.logical_and(tf.eye(shape[1], batch_shape=[shape[0]], dtype=tf.bool),
                            tf.math.logical_not(tf.cast(valid_activation_mask, tf.bool)))
        invalid_diags = tf.cast(invalid_diags, tf.float32)
        activations = activations - activations * invalid_diags
        
        # if decoder pretrain not yet done, set activations to be zero
        # to allow uniform permutation sampling
        activations = tf.cond(pretrain_done, lambda: activations, 
                              lambda: tf.add(activations, -activations * valid_activation_mask))
        
        masked_activations = activations * valid_activation_mask
        # zero-mean the activations
#         activation_avg = tf.reduce_sum(masked_activations, axis=(1,2))[:, tf.newaxis]
#         activation_avg = activation_avg / nsquare_valid
#         activations = activations - activation_avg[..., tf.newaxis] * valid_activation_mask
#         masked_activations = activations * valid_activation_mask

        # prevent the activation from being too large or too small
#         clipped_activations = tf.clip_by_value(masked_activations, -30.0, 30.0)
#         diff_activations = masked_activations - clipped_activations
#         activations = activations - diff_activations
#         masked_activations = activations * valid_activation_mask     
        
        # calculate kl divergence KL(X+eps || eps), where eps ~ gumbel noise
        kl_term1 = nsquare_valid * (tf.math.log(self.temperature) - 1.0 \
                   + np.euler_gamma * (1.0 / self.temperature - 1.0))
        s1 = 1.0 / self.temperature \
            * tf.reshape(tf.reduce_sum(masked_activations, axis=(-2,-1)), (-1,1))
        # numerical stability
        s2 = tf.reshape(tf.reduce_sum(tf.math.exp(-1.0 / self.temperature * tf.math.maximum(masked_activations, -20.0)), \
                 axis=(-2,-1)), (-1,1)) - (tf.cast(shape[1] * shape[1], tf.float32) - nsquare_valid)
        kl = kl_term1 + s1 + s2 * tf.math.exp(tf.math.lgamma(1.0 + 1.0 / self.temperature))
        #kl = kl / n_valid
        tf.print("pretrain_done", pretrain_done)
        #tf.print("masked_activations", masked_activations[0], summarize=-1)
        tf.print("kl", tf.squeeze(kl), summarize=-1)        
        kl = tf.repeat(kl, action_refinement, axis=0)
        
        
#         noise_for_log_prob = tf.stop_gradient(masked_activations + sampled_noise) - masked_activations
#         noise_for_log_prob = -(noise_for_log_prob + tf.math.exp(-noise_for_log_prob)) * valid_activation_mask
#         noise_for_log_prob = tf.reduce_sum(noise_for_log_prob, axis=(1,2))
#         noise_for_log_prob = tf.reshape(noise_for_log_prob, (-1, 1))

        # calculate normalizing constant using Bethe iteration (log denominator):
        bethe_activations = self.bethe(activations / self.temperature, **kwargs)
        #bethe_activations = self.sinkhorn(activations / self.temperature, **kwargs)
        tf.print("bethe row column sum", tf.reduce_sum(bethe_activations, axis=(-1))[0],
                                            tf.reduce_sum(bethe_activations, axis=(-2))[0],
                                            summarize=-1)        
        masked_bethe_activations = bethe_activations * valid_activation_mask
        masked_bethe_activations = tf.stop_gradient(masked_bethe_activations)
        m1mba = tf.stop_gradient((1.0 - masked_bethe_activations) * valid_activation_mask)
        eps = 1e-10
        term1 = tf.reduce_sum(masked_bethe_activations * masked_activations, 
                              axis=(-2,-1))
        term2 = -tf.reduce_sum(masked_bethe_activations * tf.math.log(masked_bethe_activations + eps), 
                               axis=(-2,-1))
        term3 = tf.reduce_sum(m1mba * tf.math.log(m1mba+eps), axis=(-2,-1))
        log_normalize_const = tf.reshape(term1 + term2 + term3, (-1, 1))
        tf.print("term1", term1[0], "term2", term2[0], "term3", term3[0])
    
#         # calculate normalizing constant using Sinkhorn iteration (log denominator):
#         sinkhorn_activations = self.sinkhorn(activations / self.temperature, **kwargs)
#         masked_sinkhorn_activations = sinkhorn_activations * valid_activation_mask
#         # prevent backprop thru sinkhorn operator
#         masked_sinkhorn_activations = tf.stop_gradient(masked_sinkhorn_activations)
#         tf.print("sinkhorn row column sum", tf.reduce_sum(sinkhorn_activations, axis=(-1))[0],
#                                             tf.reduce_sum(sinkhorn_activations, axis=(-2))[0],
#                                             summarize=-1)
#         frob_norm = tf.linalg.trace(tf.matmul(masked_activations, masked_sinkhorn_activations,
#                                               transpose_a=True))
#         eps = 1e-10
#         ent_sinkhorn_acti = -tf.reduce_sum((masked_sinkhorn_activations + eps) * tf.math.log(
#             masked_sinkhorn_activations + eps), axis=(-2,-1))
#         log_normalize_const = tf.reshape(frob_norm + ent_sinkhorn_acti, (-1, 1))
        
        
        log_normalize_const = tf.repeat(log_normalize_const, action_refinement, axis=0)
        
        # sample permutation and calculate the log nominator
        action_refinement_acti = tf.repeat(activations, action_refinement, axis=0)
        g = tfp.distributions.Gumbel(
                loc=tf.zeros_like(action_refinement_acti), 
                scale=tf.ones_like(action_refinement_acti))
        sampled_noise = g.sample()        
        action_refinement_mask = tf.repeat(valid_activation_mask, action_refinement, axis=0)
        sampled_noise = sampled_noise * action_refinement_mask      
        sample_permu = self.sinkhorn((action_refinement_acti + sampled_noise) / self.temperature, **kwargs)  
        sample_permu = tf.cast(matching(sample_permu), tf.float32)
#         tf.print("---------------------------------------")
#         tf.print("masked activations[0]", masked_activations[0], summarize=-1) 
#         tf.print("masked activations[1]", masked_activations[1], summarize=-1)         
#         tf.print("masked sinkhorn activations[0]", masked_sinkhorn_activations[0], summarize=-1) 
#         tf.print("masked sinkhorn activations[1]", masked_sinkhorn_activations[1], summarize=-1)         
#         tf.print("acti * mask[0]", action_refinement_acti[0] * action_refinement_mask[0], summarize=-1)
#         tf.print("acti * mask[1]", action_refinement_acti[1] * action_refinement_mask[1], summarize=-1)        
#         tf.print("sample_permu[0]", sample_permu[0], summarize=-1)
#         tf.print("sample_permu[1]", sample_permu[1], summarize=-1)   
#         tf.print("---------------------------------------")
        log_nominator = tf.linalg.trace(tf.matmul(action_refinement_acti * action_refinement_mask, 
                                                  sample_permu,
                                                  transpose_a=True))
        log_nominator = tf.reshape(log_nominator, (-1, 1))
        #tf.print("frob_norm", frob_norm[0], "ent_sinkhorn_acti", ent_sinkhorn_acti[0])
        tf.print("log_nominator", log_nominator[0])        
        
        # sample permutations to calculate log denominator
#         nsamples = 99
#         detach_acti = tf.stop_gradient(activations)
#         detach_acti = tf.repeat(detach_acti, nsamples, axis=0)
#         denom_g = tfp.distributions.Gumbel(
#             loc=tf.zeros_like(detach_acti), scale=tf.ones_like(detach_acti))
#         denom_noise = denom_g.sample()
#         denom_permu = self.sinkhorn((
#             detach_acti + denom_noise) / self.temperature, **kwargs) 
#         denom_permu = tf.cast(matching(denom_permu), tf.float32)
#         denom_activations = tf.repeat(masked_activations, nsamples, axis=0)
#         log_denominator = tf.linalg.trace(tf.matmul(denom_activations, denom_permu,
#                                           transpose_a=True))
#         log_denominator = tf.reshape(log_denominator, (-1, nsamples))
#         log_denominator = tf.concat([log_nominator, log_denominator], axis=1)
#         log_normalize_const = tf.math.reduce_logsumexp(log_denominator, axis=1, keepdims=True)
        
#         tf.print("log_nominator", tf.squeeze(log_nominator), 
#                  "log_denom", tf.squeeze(log_normalize_const), summarize=-1)
#         tf.print("term1", term1[0], "term2", term2[0], "term3", term3[0], 
#                  "log_nominator", log_nominator[0])
        
        # due to inexact Sinkhorn iteration, though in theory log_nominator <= log_normalize_const,
        # in practice ">" can occur
        # perdim_coeff = 1.0 / n_valid / tf.reduce_sum(1.0 / n_valid) * tf.cast(shape[0], tf.float32)
        perdim_coeff = 1.0
        return [sample_permu, action_refinement_acti * action_refinement_mask, kl * perdim_coeff, 
                log_nominator * perdim_coeff, log_normalize_const * perdim_coeff]
                #tf.math.minimum(log_nominator - log_normalize_const, 0.0)]

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
                      iterations=self.iterations,
                      temperature=self.temperature,
                      ** self.kwargs)

        base_config = super(PermutationSinkhornLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
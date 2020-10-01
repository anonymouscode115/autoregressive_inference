from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention_with_bias import AttentionWithBias
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.nn.position_encoding import position_encoding_relative
import tensorflow as tf

""" This layer is only for Permutation transformer"""
class EncoderWithPositionalAttentionLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 **kwargs):
        """Creates a Transformer encoder layer by applying a
        multi head self attention layer

        Arguments:

        input_size: int
            the number of units in the input tensor to this layer
            also the output size of the model
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
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property"""
        super(EncoderWithPositionLayer, self).__init__()

        # the core attention and processing variables
        #self.block0 = Block(hidden_size, input_size * 3, **kwargs)
        self.relative_length = 100
        self.relative_encoding = position_encoding_relative(self.relative_length,
                                                            input_size) # (range(-100,100), input_size)
        self.block0 = Block(hidden_size, input_size, lastfc=False, **kwargs)
        self.attbias0 = tf.keras.layers.Dense(heads, activation=None, **kwargs)
        self.attbias1 = tf.keras.layers.Dense(heads, activation=None, **kwargs)    
        self.q0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        self.wke0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        self.wkv0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        self.wkr0 = tf.keras.layers.Dense(input_size, activation=None, **kwargs)
        
        self.attention = AttentionWithBias(queries_dropout=queries_dropout,
                                   keys_dropout=keys_dropout,
                                   values_dropout=values_dropout,
                                   causal=causal)
        self.block1 = Block(hidden_size, input_size, **kwargs)    

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.causal = causal
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: TransformerInput
            the result of applying a multi head attention mechanism
            same shape as inputs"""

        # unpack all the requires model inputs, some might be empty tensors:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions, pointer_labels, 
         logits_labels, partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs
        
        # pass the input through a feed forward processing block and
        # separate heads from channels
        shape, dim = tf.shape(values), self.input_size // self.heads
        
        rel = pt_permutation_to_relative_l2r(1, shape[1], tf.constant(self.relative_length)) #one_hot
        rel = tf.matmul(rel, self.relative_encoding) # (1, shape[1], shape[1], input_size)
        
        x = self.block0(values, **kwargs)
        new_q = self.q0(x) # (b, shape[1], input_size)
        new_ke = self.wke0(x) # (b, shape[1], input_size)
        new_kv = self.wkv0(x) # (b, shape[1], input_size)
        new_kr = self.wkr0(rel) # (1, shape[1], shape[1], input_size)
        bias0 = self.attbias0(new_ke) # (b, shape[1], heads)
        bias1 = self.attbias1(new_kr) # (1, shape[1], shape[1], heads)
        
        def transp(u):
            return tf.transpose(tf.reshape(u, [shape[0], shape[1], self.heads, dim]),
                                [0, 2, 1, 3])
        
        new_q = transp(new_q)
        new_ke = transp(new_ke)
        new_kv = transp(new_kv)
        new_kr = tf.transpose(tf.reshape(new_kr, [1, shape[1], shape[1], self.heads, dim]),
                              [0, 3, 1, 2, 4])
        bias0 = tf.transpose(bias0, [0, 2, 1])[:, :, tf.newaxis, :]
        bias1 = tf.transpose(bias1, [0, 3, 1, 2])
        biasprod = tf.reduce_sum(new_q[:, :, :, tf.newaxis, :] * new_kr, axis=-1)
        bias = biasprod + bias0 + bias1
        
        # pass the input through an attention processing block and
        # flatten the heads and channels
        mask = tf.expand_dims(values_mask, 1)
        x = self.attention([new_q, new_ke, new_kv,
                            mask, mask, bias], **kwargs)                            
        x = tf.reshape(tf.transpose(x, [
            0, 2, 1, 3]), [shape[0], shape[1], self.heads * dim])

        # pass the outputs of the attention through another feed forward
        # processing block a residual connection
        values = values + x
        values = values + self.block1(values, **kwargs)
        
        return [queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels, 
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes]

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(input_size=self.input_size,
                      hidden_size=self.hidden_size,
                      heads=self.heads,
                      queries_dropout=self.queries_dropout,
                      keys_dropout=self.keys_dropout,
                      values_dropout=self.values_dropout,
                      causal=self.causal,
                      ** self.kwargs)

        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention import Attention
from voi.nn.base.attention_with_bias import AttentionWithBias
import tensorflow as tf


class DecoderWithPositionLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 **kwargs):
        """Creates a Transformer decoder layer by applying a
        multi head attention layer

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
        super(DecoderWithPositionLayer, self).__init__()

        # the core attention and processing variables
        self.block0 = Block(hidden_size, input_size * 3, **kwargs)
        self.pos_embedding = tf.keras.layers.Dense(input_size, **kwargs)
        self.attention0 = AttentionWithBias(queries_dropout=queries_dropout,
                                            keys_dropout=keys_dropout,
                                            values_dropout=values_dropout,
                                            causal=causal)

        # the core attention and processing variables
        self.block1 = Block(hidden_size, input_size, **kwargs)
        self.block2 = Block(hidden_size, input_size * 2, **kwargs)
        self.attention1 = Attention(queries_dropout=queries_dropout,
                                    keys_dropout=keys_dropout,
                                    values_dropout=values_dropout,
                                    causal=False)
        self.block3 = Block(hidden_size, input_size, **kwargs)

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

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        s0, s1 = tf.shape(queries), tf.shape(values)
        dim = self.input_size // self.heads

        # pass the input through a feed forward processing block and
        # separate heads from channels
        x = self.block0(queries, **kwargs)
        x = tf.transpose(tf.reshape(x, [
            s0[0], s0[1], self.heads, dim * 3]), [0, 2, 1, 3])

        # add a position-conditioned bias to the attention scores
        # in log-space: https://arxiv.org/pdf/1902.01370.pdf
        pos = self.pos_embedding(relative_positions, **kwargs)
        pos = tf.transpose(tf.reshape(pos, [
            s0[0], s0[1], s0[1], self.heads, dim]), [0, 3, 1, 2, 4])
        bias = tf.squeeze(tf.matmul(
            tf.expand_dims(x[..., :dim], 3), pos, transpose_b=True), 3)

        # pass the input through an attention processing block and
        # flatten the heads and channels
        mask0 = tf.expand_dims(queries_mask, 1)
        x = self.attention0([x[..., :dim], x[..., dim:2*dim], x[..., 2*dim:],
                             mask0, mask0, bias], **kwargs)
        x = tf.reshape(tf.transpose(x, [
            0, 2, 1, 3]), [s0[0], s0[1], self.heads * dim])

        # pass the input through a feed forward processing block and
        # separate heads from channels
        queries = queries + x
        y = self.block1(queries, **kwargs)
        y = tf.transpose(tf.reshape(y, [
            s0[0], s0[1], self.heads, dim]), [0, 2, 1, 3])

        # pass the input through a feed forward processing block and
        # separate heads from channels
        x = self.block2(values, **kwargs)
        x = tf.transpose(tf.reshape(x, [
            s1[0], s1[1], self.heads, dim * 2]), [0, 2, 1, 3])

        # pass the input through an attention processing block and
        # flatten the heads and channels
        mask1 = tf.expand_dims(values_mask, 1)
        x = self.attention1([y, x[..., :dim], x[..., dim:],
                             mask0, mask1], **kwargs)
        x = tf.reshape(tf.transpose(x, [
            0, 2, 1, 3]), [s0[0], s0[1], self.heads * dim])

        # pass the outputs of the attention through another feed forward
        # processing block a residual connection
        queries = queries + x
        queries = queries + self.block3(queries, **kwargs)
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

        base_config = super(DecoderWithPositionLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

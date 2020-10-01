import tensorflow as tf


class SequenceToMatSinkhorn(tf.keras.layers.Layer):

    def __init__(self,
                 queries_dropout=0.,
                 keys_dropout=0.):
        """Creates the backbone for the logits of a permutation layer
        converts a sequence to a matrix of permutation logits
        Arguments:
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer"""
        super(SequenceToMatSinkhorn, self).__init__()

        self.q_dropout = tf.keras.layers.Dropout(queries_dropout)
        self.k_dropout = tf.keras.layers.Dropout(keys_dropout)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.queries_dropout_rate = queries_dropout
        self.keys_dropout_rate = keys_dropout

    def static_call(self, queries, keys,
                    queries_mask, values_mask, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput
        Arguments:
        inputs: AttentionInput
            a dataclass instance that contains queries, keys
            and values along with masks
        Returns:
        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        # apply dropout to the queries keys and values tensor
        # requires all to be like [batch, heads, ]
        queries = self.q_dropout(queries, **kwargs)
        keys = self.k_dropout(keys, **kwargs)

        # compute the multi head soft attention weights using
        # scaled dot product attention
        size = tf.math.sqrt(
            tf.cast(tf.shape(queries)[-1], tf.float32))
        scores = tf.matmul(
            queries, keys, transpose_b=True) / size

        # if an attention bias is provided that add the attention bias
        # to the pre softmax scores matrix
#         if hasattr(inputs, 'bias') and inputs.bias is not None:
#             scores = scores + inputs.bias

        # apply a mask to the scores matrix so that only real
        # non terminal elements are permuted out of place
        mask = tf.expand_dims(values_mask, -2)
        mask = tf.logical_and(mask, tf.expand_dims(queries_mask, -1))

        # pad tokens should not be permuted and logits on the diagonal
        # for pad tokens should not be masked out; this is necessary because
        # a valid permutation matrix has rows and columns that sum to one,
        # even for rows that correspond to pad tokens
        shape = tf.shape(mask)
        mask = tf.logical_or(mask, tf.eye(
            shape[-2],
            num_columns=shape[-1], batch_shape=shape[:-2], dtype=tf.bool))

        # apply a boolean mask to the keys and values
        return tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.))

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput

        Arguments:

        inputs: AttentionInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        return self.static_call(inputs.queries, inputs.keys,
                                inputs.queries_mask, inputs.values_mask,
                                **kwargs)
    
    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process
        Returns:
        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(queries_dropout=self.queries_dropout_rate,
                      keys_dropout=self.keys_dropout_rate)

        base_config = super(SequenceToMatSinkhorn, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
from voi.nn.base.block import Block
from voi.nn.base.attention import causal_mask
from voi.nn.variables.pointer import Pointer
import tensorflow as tf


class PointerAfterLogits(Pointer):

    def __init__(self,
                 hidden_size,
                 output_size,
                 logits_size,
                 logits_embedding,
                 causal=True,
                 logits_per_slot=1,
                 **kwargs):
        """Creates a pointer network using the first operation
        in the self attention mechanism

        Arguments:

        hidden_size: int
            the number of hidden units in the network blocks
            used by this layer
        output_size: int
            the number of output units used by the network blocks
            used by this layer
        logits_size: int
            the number of units in the vector space of the logits
            of a transformer model
        logits_embedding: tf.keras.layers.Embedding
            the shared embedding matrix for word and pointer 
            prediction               
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property
        logits_per_slot: int
            specifies the number of logits per element the pointer
            network attends to; default is 1"""
        super(PointerAfterLogits, self).__init__(
            hidden_size,
            output_size,
            causal=causal,
            logits_per_slot=logits_per_slot,
            **kwargs)

        # the core processing variables
        self.logits_embedding = logits_embedding

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.logits_size = logits_size

    def call(self, inputs, **kwargs):
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        pointer_logits: tf.Tensor
            the logits of a pointer network used to select locations to
            insert words in a transformer
        target_left: tf.Tensor
            the unsorted position of the token to the left, in terms of 
            sorted position, of a certain token, at each decoding time step
            (time steps 0 to T-1)
        target_right: tf.Tensor
            the unsorted position of the token to the right, in terms of 
            sorted position, of a certain token, at each decoding time step"""

        # unpack all the requires model inputs, some might be empty tensors:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs

        # map the sequence into a latent space
        features = self.block(queries, **kwargs)
        q = features[..., :self.output_size]
        
        embedding_before_permutation = self.logits_embedding(ids)
        if permutation != None:
            q = q + tf.matmul(permutation[:, 1:, 1:],
                              embedding_before_permutation)
        else:
            q = q + embedding_before_permutation        

        # reshape keys to have logits_per_slot more time steps
        shape = tf.multiply(tf.shape(q), [1, self.logits_per_slot, 1])
        k = tf.reshape(features[..., self.output_size:], shape)
        size = tf.math.sqrt(tf.cast(tf.shape(q)[2], tf.float32))
        
        # Reorder keys such that inserting to the left corresponds to 
        # matmul with the (projected) hidden vector of the left token 
        # in absolute position. Similar does inserting to the right
        valid_range = tf.range(tf.shape(partial_pos)[-1]) + 1
        valid_range = valid_range[tf.newaxis, tf.newaxis, :, tf.newaxis]
        target_left = tf.math.floormod(partial_pos - 1, valid_range)
        target_right = tf.math.floormod(partial_pos + 1, valid_range)
        mask = tf.linalg.band_part(tf.ones_like(partial_pos), -1, 0)
        mask = tf.cast(mask, tf.bool)
        partial_pos = tf.where(mask, partial_pos, 999999)
        target_left = tf.where(mask, target_left, 999999)
        target_right = tf.where(mask, target_right, 999999)
        target_left = tf.math.argmin(
            tf.abs(partial_pos[:, :, :, tf.newaxis, :]
                   - target_left[:, :, :, :, tf.newaxis]
                  ), 
            axis=-1,
            output_type=tf.int32
        )             
        target_right = tf.math.argmin(
            tf.abs(partial_pos[:, :, :, tf.newaxis, :]
                   - target_right[:, :, :, :, tf.newaxis]
                  ), 
            axis=-1,
            output_type=tf.int32
        )  
        
        # calculate raw logit for scores and reorder according to target_left and 
        # target_right
        raw_logits = tf.matmul(q, k, transpose_b=True) / size
        #tf.print("raw_logits", tf.transpose(raw_logits[0][:,:6], [0,1]), summarize=-1)
        
        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(queries_mask, 2),
                              tf.expand_dims(queries_mask, 1))
        if self.causal:
            cmsk = causal_mask(raw_logits[:, tf.newaxis, :, ::self.logits_per_slot])
            mask = tf.logical_and(mask, tf.squeeze(cmsk, 1))

        # filter by removing logits for elements that are invalid
        # mask must be repeated to correct the shape
        mask = tf.repeat(mask, self.logits_per_slot, axis=2)
        return (
            tf.where(mask, raw_logits, 
                     tf.fill(tf.shape(raw_logits), -999999.)
                    ),
            target_left, 
            target_right
        )

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      output_size=self.output_size,
                      logits_size=self.logits_size,
                      logits_embedding=self.logits_embedding,
                      causal=self.causal,
                      logits_per_slot=self.logits_per_slot,
                      **self.kwargs)

        base_config = super(PointerAfterLogits, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

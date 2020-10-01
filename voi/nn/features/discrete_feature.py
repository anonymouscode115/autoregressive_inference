from voi.nn.wrappers.layer import Layer
from voi.nn.position_encoding import position_encoding
import tensorflow as tf


class DiscreteFeature(Layer):

    def __init__(self,
                 hidden_size,
                 src_embedding,
                 tgt_embedding,
                 mode='decoder',
                 decoder_pos_emb=False,
                 **kwargs):
        """Creates a Transformer embedding layer by applying a
        lookup operation to the queries

        Arguments:

        num_embeddings: int
            the number of elements in the vocabulary which
            input sequences contain elements of
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        src_embedding: tf.keras.layers.Embedding
                    embedding of source vocabulary           
        tgt_embedding: tf.keras.layers.Embedding
                    embedding of target vocabulary
        decoder_pos_emb: bool
            whether to add positional embedding to the decoder to let it know
            its own generation ordering
        mode: str
            decoder or permutation transformer"""
        super(DiscreteFeature, self).__init__()

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.mode = mode
        self.decoder_pos_emb = decoder_pos_emb
        self.kwargs = kwargs

        # the core processing variables
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding

    def call(self, inputs, **kwargs):
        """Runs a forward pass on the embeddings of a transformer
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

        a = position_encoding(tf.shape(queries)[1], self.hidden_size)
        b = self.tgt_embedding(queries, **kwargs)
        if self.mode == 'decoder':
            b = tf.matmul(absolute_positions, b)
            if self.decoder_pos_emb:
                b = a + b
        elif self.mode == 'pt' and self.decoder_pos_emb:
            # we do need positional encoding for Permutation Transformer
            b = a + b            
        c = position_encoding(tf.shape(values)[1], self.hidden_size)
        d = self.src_embedding(values, **kwargs)
        if self.mode == 'decoder':
            if self.decoder_pos_emb:
                d = c + d
        elif self.mode == 'pt' and self.decoder_pos_emb:
            d = c + d
            
        return  [b, d, queries_mask, values_mask, ids, permutation,
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
        config = dict(hidden_size=self.hidden_size,
                      src_embedding=self.src_embedding,
                      tgt_embedding=self.tgt_embedding,
                      mode=self.mode,
                      decoder_pos_emb=self.decoder_pos_emb,
                      ** self.kwargs)

        base_config = super(DiscreteFeature, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

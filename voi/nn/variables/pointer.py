from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
from voi.nn.base.attention import causal_mask
import tensorflow as tf
import tree

# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32)])
def custom_gather(a, b):
    """ Permutes the vectors in the last dimension of a according to
        the corresponding vectors in the last dimension of b
        e.g. a=[[1,4,7,6],[4,3,6,8]], b=[[0,3,1,2],[2,1,3,0]]
        Returns [[1,6,4,7],[6,3,8,4]]
    """
    assert len(tf.shape(a)) >= 2 # first dimension is batch size
    assert len(tf.shape(a)) == len(tf.shape(b))
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

# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
def partial_pos_from_hard_relative(R):
    """ Calculates partial position matrix given hard relative position matrix;
        this function is only called during greedy/beam search because, during training,
        the relative position matrix can be soft
    """
    assert len(tf.shape(R)) == 4
    R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
    partial_pos = tf.repeat(
        R[:, tf.newaxis, :, :], tf.shape(R)[-1], axis=-3)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 3, 2, 1]), 0, -1)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 2, 1, 3]), 0, -1)
    partial_pos = tf.cast(tf.reduce_sum(tf.maximum(
        0, tf.transpose(partial_pos, [0, 3, 1, 2])), axis=-2), tf.int32)
    return partial_pos[:, tf.newaxis, :, :]    

# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32),
#     tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32)])
def prob_both_insertion(raw_logits, 
                        target_left, target_right):
    """ Returns the total probability of inserting at a token's right,
        by handling the case that inserting to both left and right is allowed
        i.e. add the probability of inserting at a token's left
        to the probability of inserting at the corresponding
        token's right; the returned value is passed through
        cross entropy with inputs.right_pointer_labels to calculate loss
    """
    raw_logits = raw_logits[:, tf.newaxis, :, :]                            
    probs = tf.math.softmax(raw_logits, axis=-1)   
    probs_left = probs[..., ::2]
    probs_right = probs[..., 1::2]
    probs_left = custom_gather(probs_left, target_right)
#         tf.print("probs left", probs_left[0,0], summarize=-1)
#         tf.print("probs right", probs_right[0,0], summarize=-1)
    probs = probs_left + probs_right
    probs = tf.squeeze(probs, 1)          
    return probs


class Pointer(Layer):

    def __init__(self,
                 hidden_size,
                 output_size,
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
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property
        logits_per_slot: int
            specifies the number of logits per element the pointer
            network attends to; default is 1"""
        super(Pointer, self).__init__()

        # the core processing variables
        self.block = Block(
            hidden_size, output_size * (1 + logits_per_slot), **kwargs)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.causal = causal
        self.logits_per_slot = logits_per_slot
        self.kwargs = kwargs
    
    def call(self, inputs, **kwargs):
        """Use the call function of classes that inherit Pointer,
        e.g. in pointer_after_logits.py """

        raise NotImplementedError

    def loss(self, inputs, **kwargs):
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        loss: tf.Tensor
            a loss function that computes the contribution this layer
            makes to the total model loss
        outputs: tf.Tensor
            the logits of a transformer model used for word prediction
            or a pointer network"""

        # unpack all the requires model inputs, some might be empty tensors:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs
        
        if self.logits_per_slot > 2:
            raise NotImplementedError("logits per slots > 2 not implemented yet")
            
        pointer, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(pointer, target_left, target_right)
        loss = tf.keras.losses.categorical_crossentropy(
                   pointer_labels, probs, from_logits=False)
        pointer_probs = -loss
        return loss, [queries, values, queries_mask, values_mask, ids, permutation,
             absolute_positions, relative_positions, pointer_labels, 
             logits_labels, partial_pos, pointer_probs, log_probs,
             object_detections, object_features, object_boxes]

    def greedy_search(self,
                      inputs,
                      closed,
                      **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified"""

        # unpack all the requires model inputs, some might be empty tensors:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs
        
        # compute a distribution over tokens; note that only one token
        # is sampled yet top_k is a convenient function        
        raw_logits, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(raw_logits, target_left, target_right)
        logits = tf.math.log(probs[:, -1, :] + 1e-7)            
        _log_probs, _ids = tf.math.top_k(logits, k=1)        

        # calculate the position of the rightmost token
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        rightmost_ids = tf.argmax(tf.reduce_sum(
            tf.nn.relu(R), axis=-2), axis=-1, output_type=tf.int32)

        # mask the log probabilities and tokens of already completed
        # beams so that they are unchanged when decoding
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, tf.zeros_like(_log_probs), _log_probs)
        _ids = tf.where(mask, rightmost_ids[:, tf.newaxis], _ids)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        r = tf.gather(R, _ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)
        partial_pos = partial_pos_from_hard_relative(relative_positions)

        # compute the update log probability and note that the pointer network
        # does not specify a termination condition by itself
        log_probs = log_probs + _log_probs[..., 0]
        return ([queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels, 
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes], closed)

    def beam_search(self,
                    inputs,
                    closed,
                    last_beam_size,
                    beam_size,
                    **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        last_beam_size: int
            the number of beams that were expanded by the last layer in an
            autoregressive model
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model"""

        # unpack all the requires model inputs, some might be empty tensors:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs
        # compute a distribution over pointer locations
        raw_logits, target_left, target_right = self.call(inputs, **kwargs)
        probs = prob_both_insertion(raw_logits, target_left, target_right)
        logits = tf.math.log(probs[:, -1, :] + 1e-7)
        
        batch_size = tf.shape(logits)[0] // last_beam_size

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        sample_size = tf.minimum(tf.shape(logits)[1], beam_size)

        # sample the top beam_size candidates
        _log_probs, _ids = tf.math.top_k(logits, k=sample_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(_log_probs)[:1], 0), sample_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # calculate the position of the rightmost token
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        rightmost_ids = tf.argmax(tf.reduce_sum(
            tf.nn.relu(R), axis=-2), axis=-1, output_type=tf.int32)

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, closed_log_probs, _log_probs)
        _ids = tf.where(mask, rightmost_ids[:, tf.newaxis], _ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size, sample_size])
        _log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, 1]) + _log_probs
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size * sample_size])

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        cand_size = tf.minimum(tf.shape(_log_probs)[1], beam_size)

        # select the top beam_size candidates
        _log_probs, beam_ids = tf.math.top_k(_log_probs, k=cand_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        old_beam_ids = tf.math.floordiv(beam_ids, sample_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        _ids = tf.reshape(_ids, [batch_size, last_beam_size * sample_size])
        _ids = tf.gather(_ids, beam_ids, batch_dims=1)
        _ids = tf.reshape(_ids, [batch_size * cand_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            if x is None:
                return x
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * cand_size], shape], axis=0)
            return tf.reshape(tf.gather(
                tf.reshape(x, s0), old_beam_ids, batch_dims=1), s1)

        # select which old beams are propagated forward
        # this is necessary because some beams have content-aware state
        queries = select(queries)
        values = select(values)
        queries_mask = select(queries_mask)
        values_mask = select(values_mask)
        ids = select(ids)
        permutation = select(permutation)
        absolute_positions = select(absolute_positions)
        relative_positions = select(relative_positions)
        partial_pos = select(partial_pos)
        pointer_labels = select(pointer_labels)
        logits_labels = select(logits_labels)

        # TODO: Brandon -> handle the image features as well.
        object_detections = select(object_detections)
        object_features = select(object_features)
        object_boxes = select(object_boxes)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        R = relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        r = tf.gather(R, _ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)
        partial_pos = partial_pos_from_hard_relative(relative_positions)

        # update log probability and note that the pointer network
        # does not specify a termination condition by itself
        log_probs = tf.reshape(_log_probs, [batch_size * cand_size])
        return ([queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels, 
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes],
                select(closed), cand_size)

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
                      causal=self.causal,
                      logits_per_slot=self.logits_per_slot,
                      **self.kwargs)

        base_config = super(Pointer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

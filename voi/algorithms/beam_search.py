import tensorflow as tf


def beam_search(inputs,
                model,
                dataset, 
                beam_size=8,
                max_iterations=20,
                return_rel_pos=False):
    """Perform a beam search using a transformer model that accepts
    region features as input

    Arguments:

    inputs: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a layers model that accepts inputs in the form of the dataclass
        TransformerInput and returns logits
    dataset: str
        the type of dataset (captioning or wmt)
    beam_size: int
        the number of beams to use when calculating a beam search
        a beam size of zero is a greedy search
    max_iterations: int
        the maximum number of decoding steps to use when performing
        beam search; the maximum sequence length
    return_rel_pos: bool
        whether to return relative position matrix (for inspect_order.py)

    Returns:

    sequence: tf.Tensor
        a tensor that contains output word ids that were taken
        when decoding using the transformer
    log_p: tf.Tensor
        the log probability of predicted sentences under the
        current transformer model"""

    # unpack all the requires model inputs:
    [queries, values, queries_mask, values_mask, ids, permutation,
     absolute_positions, relative_positions,
     pointer_labels, logits_labels, _, _, log_probs,
     object_detections, object_features, object_boxes] = inputs

    # meta data to keep track of which beams have completed
    # during the decoding step
    batch_size = tf.shape(values_mask)[0]
    closed = tf.fill([batch_size], False)
    last_beam_size = tf.constant(1)

    # replace the model inputs with an empty sentence that will be
    # appended to during the decoding step
    queries = tf.fill([batch_size, 1], 2)
    queries_mask = tf.fill([batch_size, 1], True)
    
    old_values = values
    if dataset == 'captioning':
        # dummy value variable
        values = queries
        
    ids = tf.fill([batch_size, 0], 2)
    relative_positions = tf.one_hot(tf.fill([batch_size, 1, 1], 1), 3)
    absolute_positions = tf.eye(1, batch_shape=[batch_size])
    partial_pos = tf.zeros([batch_size, 1, 1, 1], dtype=tf.int32)
    permutation = tf.eye(2, batch_shape=[batch_size])
    pointer_labels = tf.zeros([batch_size])
    logits_labels = tf.zeros([batch_size])
    log_probs = tf.zeros([batch_size])
    pointer_probs = tf.zeros([batch_size])    

    def update(queries, values, queries_mask, values_mask, ids, permutation,
               absolute_positions, relative_positions,
               pointer_labels, logits_labels, 
               partial_pos, pointer_probs, log_probs,
               object_detections, object_features, object_boxes,
               closed, last_beam_size, i):

        # format the inputs for the transformer in the next round
        inputs, closed, last_beam_size = model.beam_search(
            [queries, values, queries_mask, values_mask, ids, permutation,
             absolute_positions, relative_positions,
             pointer_labels, logits_labels, 
             partial_pos, pointer_probs, log_probs,
             object_detections, object_features, object_boxes],
            closed, last_beam_size, beam_size)

        # unpack all the requires model inputs:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs

        # we need to replace the transformer inputs at every
        # iteration of decoding
        start = tf.fill([batch_size * last_beam_size, 1], 2)
        absolute_positions = tf.eye(2 + i, batch_shape=[batch_size * last_beam_size])
        permutation = tf.eye(3 + i, batch_shape=[batch_size * last_beam_size])
        queries = tf.concat([start, ids], axis=1)
        queries_mask = tf.concat([
            queries_mask,
            tf.logical_not(closed)[:, tf.newaxis]], axis=1)
        if dataset == 'captioning':
            values = queries
        elif dataset in ['wmt', 'django', 'gigaword']:
            values = tf.repeat(old_values, last_beam_size, axis=0)

        i = i + 1
        closed = tf.logical_or(closed, tf.greater_equal(i, max_iterations))
        return [queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels, 
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes,
                closed, last_beam_size, i]

    def cond(queries, values, queries_mask, values_mask, ids, permutation,
             absolute_positions, relative_positions,
             pointer_labels, logits_labels, 
             partial_pos, pointer_probs, log_probs,
             object_detections, object_features, object_boxes,
             closed, last_beam_size, i):

        return tf.logical_not(tf.reduce_all(closed))

    # loop for a maximum of max_iterations decoding steps
    i = tf.constant(0)

    outputs = tf.while_loop(
        cond,
        update,
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes,
         closed, last_beam_size, i],
        shape_invariants=[
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None, None, 3]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None, 4]),
            tf.TensorShape([None]),
            last_beam_size.get_shape(),
            i.get_shape()])

    [queries, values, queries_mask, values_mask, ids, permutation,
     absolute_positions, relative_positions,
     pointer_labels, logits_labels, 
     partial_pos, pointer_probs, log_probs,
     object_detections, object_features, object_boxes,
     closed, last_beam_size, i] = outputs

    # helper function for un flattening the beam size from the batch axis
    def expand(x):
        return tf.reshape(x, tf.concat([[
            batch_size, last_beam_size], tf.shape(x)[1:]], axis=0))

    # decoding is finished so un flatten the beam dimension
    # returns a shape like [batch_size, beam_size, sequence_length]
    ids = expand(ids)

    # when the model decodes permutation matrices in additions to ids;
    # then sort ids according to the decoded permutation
    if model.final_layer == 'indigo':
        pos = relative_positions
        pos = tf.argmax(pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(expand(pos[:, 1:, 1:])), axis=2)
        pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.int32)
        ids = tf.squeeze(tf.matmul(tf.expand_dims(ids, 2), pos), 2)

    if not return_rel_pos:
        return ids, tf.reshape(
            log_probs, [batch_size, last_beam_size])
    else:
        return ids, tf.reshape(
            log_probs, [batch_size, last_beam_size]), expand(relative_positions)

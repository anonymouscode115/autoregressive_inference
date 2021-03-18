import tensorflow as tf


def nucleus_sampling(inputs,
                     model,
                     dataset,
                     max_iterations=20,
                     nucleus_probability=0.95,
                     num_samples=5,
                     return_rel_pos=False):
    """Decode a sequence of tokens using the nucleus sampling
    strategy, and return several independent samples

    Arguments:

    inputs: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a layers model that accepts inputs in the form of the dataclass
        TransformerInput and returns logits
    dataset: str
        the type of dataset (captioning or wmt)
    max_iterations: int
        the maximum number of decoding steps to use when performing
        greedy search; the maximum sequence length
    nucleus_probability: float
        the probability threshold used to determine the size
        of the nucleus set of tokens to sample from
    num_samples: int
        the number of independent identically distributed samples
        to draw from the probability distribution given by the nucleus
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

    # replace the model inputs with an empty sentence that will be
    # appended to during the decoding step
    start = tf.fill([batch_size, 1], 2)
    queries = start
    queries_mask = tf.fill([batch_size, 1], True)

    old_values = values
    if dataset == 'captioning':
        # dummy value variable
        values = queries

    ids = tf.fill([batch_size, 0], 2)
    partial_pos = tf.zeros([batch_size, 1, 1, 1], dtype=tf.int32)
    relative_positions = tf.one_hot(tf.fill([batch_size, 1, 1], 1), 3)
    absolute_positions = tf.eye(1, batch_shape=[batch_size])
    permutation = tf.eye(2, batch_shape=[batch_size])
    pointer_labels = tf.zeros([batch_size])
    logits_labels = tf.zeros([batch_size])
    log_probs = tf.zeros([batch_size])
    pointer_probs = tf.zeros([batch_size])

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    queries = tf.repeat(queries, num_samples, axis=0)
    values = tf.repeat(values, num_samples, axis=0)
    queries_mask = tf.repeat(queries_mask, num_samples, axis=0)
    values_mask = tf.repeat(values_mask, num_samples, axis=0)
    ids = tf.repeat(ids, num_samples, axis=0)

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    partial_pos = tf.repeat(partial_pos, num_samples, axis=0)
    pointer_probs = tf.repeat(pointer_probs, num_samples, axis=0)
    log_probs = tf.repeat(log_probs, num_samples, axis=0)

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    relative_positions = tf.repeat(relative_positions, num_samples, axis=0)
    absolute_positions = tf.repeat(absolute_positions, num_samples, axis=0)
    permutation = tf.repeat(permutation, num_samples, axis=0)
    pointer_labels = tf.repeat(pointer_labels, num_samples, axis=0)
    logits_labels = tf.repeat(logits_labels, num_samples, axis=0)

    # duplicate the features by the number of samples to draw from
    # the nucleus sampling distribution: [batch, num_samples]
    object_detections = tf.repeat(object_detections, num_samples, axis=0)
    object_features = tf.repeat(object_features, num_samples, axis=0)
    object_boxes = tf.repeat(object_boxes, num_samples, axis=0)
    closed = tf.repeat(closed, num_samples, axis=0)

    def update(queries, values, queries_mask, values_mask, ids, permutation,
               absolute_positions, relative_positions,
               pointer_labels, logits_labels,
               partial_pos, pointer_probs, log_probs,
               object_detections, object_features, object_boxes,
               closed, i):

        # decode using the model for a single pass
        inputs, closed = model.nucleus_sampling(
            [queries, values, queries_mask, values_mask, ids, permutation,
             absolute_positions, relative_positions,
             pointer_labels, logits_labels,
             partial_pos, pointer_probs, log_probs,
             object_detections, object_features, object_boxes],
            closed, nucleus_probability)

        # unpack all the requires model inputs:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels,
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs

        # we need to replace the transformer inputs at every
        # iteration of decoding
        start = tf.fill([batch_size * num_samples, 1], 2)
        absolute_positions = tf.eye(2 + i, batch_shape=[batch_size * num_samples])
        permutation = tf.eye(3 + i, batch_shape=[batch_size * num_samples])
        queries = tf.concat([start, ids], axis=1)
        queries_mask = tf.concat([
            queries_mask,
            tf.logical_not(closed)[:, tf.newaxis]], axis=1)
        if dataset == 'captioning':
            values = queries
        elif dataset in ['wmt', 'django', 'gigaword']:
            values = tf.repeat(old_values, num_samples, axis=0)

        i = i + 1
        closed = tf.logical_or(closed, tf.greater_equal(i, max_iterations))
        return [queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels,
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes,
                closed, i]

    def cond(queries, values, queries_mask, values_mask, ids, permutation,
             absolute_positions, relative_positions,
             pointer_labels, logits_labels,
             partial_pos, pointer_probs, log_probs,
             object_detections, object_features, object_boxes,
             closed, i):

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
         closed, i],
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
            i.get_shape()])

    [queries, values, queries_mask, values_mask, ids, permutation,
     absolute_positions, relative_positions,
     pointer_labels, logits_labels,
     partial_pos, pointer_probs, log_probs,
     object_detections, object_features, object_boxes,
     closed, i] = outputs

    # helper function for un flattening the beam size from the batch axis
    def expand(x):
        return tf.reshape(x, tf.concat([[
            batch_size, num_samples], tf.shape(x)[1:]], axis=0))

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
            log_probs, [batch_size, num_samples])
    else:
        return ids, tf.reshape(
            log_probs, [batch_size, num_samples]), expand(relative_positions)

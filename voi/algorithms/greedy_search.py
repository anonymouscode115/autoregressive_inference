import tensorflow as tf


def greedy_search(inputs,
                  model,
                  max_iterations=20):
    """Perform a greedy search using a transformer model that
    accepts region features as input

    Arguments:

    inputs: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a layers model that accepts inputs in the form of the dataclass
        TransformerInput and returns logits
    max_iterations: int
        the maximum number of decoding steps to use when performing
        greedy search; the maximum sequence length

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
    ids = tf.fill([batch_size, 0], 2)
    partial_pos = tf.zeros([batch_size, 1, 1, 1], dtype=tf.int32)
    relative_positions = tf.one_hot(tf.fill([batch_size, 1, 1], 1), 3)
    absolute_positions = tf.eye(1, batch_shape=[batch_size])
    log_probs = tf.zeros([batch_size])
    pointer_probs = tf.zeros([batch_size])
    
    # loop for a maximum of max_iterations decoding steps
    for i in range(max_iterations):

        # exit if all beams have finished decoding
        if tf.reduce_all(closed):
            break

        # decode using the model for a single pass
        inputs, closed = model.greedy_search(
            [queries, values, queries_mask, values_mask, ids, permutation,
             absolute_positions, relative_positions,
             pointer_labels, logits_labels, 
             partial_pos, pointer_probs, log_probs,
             object_detections, object_features, object_boxes], closed)

        # unpack all the requires model inputs:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs

        # the transformer modifies in place the input data class so
        # we need to replace the transformer inputs at every
        # iteration of decoding
        queries = tf.concat([start, ids], axis=1)
        queries_mask = tf.concat([
            inputs.queries_mask,
            tf.logical_not(closed)[:, tf.newaxis]], axis=1)

    # when the model decodes permutation matrices in additions to ids;
    # then sort ids according to the decoded permutation
    if model.final_layer == 'indigo':
        pos = relative_positions
        pos = tf.argmax(pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(pos[:, 1:, 1:]), axis=1)
        pos = tf.one_hot(pos, tf.shape(pos)[1], dtype=tf.int32)
        ids = tf.squeeze(
            tf.matmul(tf.expand_dims(ids, 1), pos), 1)

    # unlike beam search we can directly return the result without
    # calling reshape since there is not an extra axis
    return ids, log_probs

from voi.nn.wrappers.layer import Layer
import tensorflow as tf
import tensorflow_probability as tfp
import tree


class Logits(Layer):

    def __init__(self,
                 output_size,
                 logits_embedding,
                 label_smoothing=0.0,
                 **kwargs):
        """Creates a Transformer embedding layer by applying a
        lookup operation to the queries

        Arguments:

        output_size: int
            the number of units in the vector space of the logits
            of a transformer model
        logits_embedding: tf.keras.layers.Embedding
            the shared embedding matrix for word and pointer 
            prediction
        label_smoothing: float
            coefficient for label smoothing"""
        
        super(Logits, self).__init__()

        # the core processing variables
        self.dense = tf.keras.layers.Dense(output_size, **kwargs)
        self.logits_embedding = logits_embedding
        self.label_smoothing = label_smoothing
        
        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.output_size = output_size
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):

        """ 
        This is a dummy function. 
        Before we train the model,
        we "call" the autoregressive decoder only to initialize 
        the parameters.
        During training, the function we invoke is self.loss.
        During greedy/beam search, the function we invoke is 
        self.logits_before_softmax.
        In either case, we don't invoke model(inputs),
        which passes the inputs through the "call" function
        of every module of the decoder.
        """
        
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, 
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs
        
        return inputs
    
    def logits_before_softmax(self, queries, **kwargs):
        
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput
        Arguments:
        queries: tf.Tensor
            inputs.queries
        Returns:
        outputs: tf.Tensor
            the logits of a transformer model used for word prediction
            or a pointer network"""
        
        result = tf.matmul(self.dense(queries, **kwargs),
                           self.logits_embedding.weights,
                           transpose_b = True)
        return result    

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

        logits = self.logits_before_softmax(queries, **kwargs)
        num_classes = tf.cast(tf.shape(logits_labels)[-1], tf.float32)
        logits_labels = tf.cast(logits_labels, tf.float32)
        logits_labels = logits_labels * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)   
#         log_probs = tf.nn.log_softmax(logits, axis=-1)
#         eps = 1e-8
#         log_truth = tf.math.log(logits_labels + eps)
#         return tf.reduce_sum(logits_labels * (log_truth - log_probs), axis=-1), inputs
        return tf.keras.losses.categorical_crossentropy(
            logits_labels, logits, from_logits=True), inputs

    def greedy_search(self,
                      inputs,
                      closed,
                      **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        inputs: Dataclass
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
        logits = tf.math.log_softmax(self.logits_before_softmax(
                                     queries, **kwargs)[:, -1])
        _log_probs, _ids = tf.math.top_k(logits, k=1)

        # mask the log probabilities and tokens of already completed
        # beams so that they are unchanged when decoding
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, tf.zeros_like(_log_probs), _log_probs)
        _ids = tf.where(mask, tf.zeros_like(_ids), _ids)

        # concatenate the sampled tokens to the beam and prepare the
        # model outputs for the next layer; also compute if we
        # has finished decoding by predicting the end token
        ids = tf.concat([ids, _ids], 1)
        log_probs = log_probs + _log_probs[..., 0]
        return ([queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels, 
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes],
                tf.logical_or(closed, tf.equal(_ids[:, 0], 3)))

    def nucleus_sampling(self,
                         inputs,
                         closed,
                         nucleus_probability,
                         **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using the nucleas sampling strategy

        Arguments:

        inputs: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        nucleus_probability: float
            the probability threshold used to determine the size
            of the nucleus set of tokens to sample from

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
        logits = tf.math.log_softmax(
            self.logits_before_softmax(queries, **kwargs)[:, -1])

        # build an objective to determine the optimal value of k
        sorted_logits = tf.sort(logits, axis=-1, direction='DESCENDING')
        set_probs = tf.math.cumsum(tf.exp(sorted_logits), axis=-1)
        objective = tf.abs(set_probs - nucleus_probability) + \
                    999999.0 * tf.cast(
            set_probs < nucleus_probability, tf.float32)

        # an adaptive value of k that is different for different
        # elements in a batch, for nucleas sampling
        k = tf.math.argmin(objective, axis=-1, output_type=tf.int32)
        min_logits = tf.gather(sorted_logits, k[..., tf.newaxis], batch_dims=1)

        # build a distribution to sample from using the selected k
        # we take the maximum k so that each batch can
        # be processed at the same time
        _log_probs, _ids = tf.math.top_k(logits, k=tf.reduce_max(k) + 1)

        # mask the log probs to remove tokens with low probability
        # then normalize the distribution by subtracting the log denominator
        _log_probs -= 999999.0 * tf.cast(_log_probs < min_logits, tf.float32)
        _log_probs = tf.math.log_softmax(_log_probs)

        # sample from the probability distribution represented
        # by the nucleas set of tokens
        dist = tfp.distributions.Categorical(logits=_log_probs)
        nucleus_samples = dist.sample()[..., tf.newaxis]

        # aggregate the samples according to which token from
        # the nucleas was selected
        _log_probs = tf.gather(_log_probs, nucleus_samples, batch_dims=1)
        _ids = tf.gather(_ids, nucleus_samples, batch_dims=1)

        # mask the log probabilities and tokens of already completed
        # beams so that they are unchanged when decoding
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, tf.zeros_like(_log_probs), _log_probs)
        _ids = tf.where(mask, tf.zeros_like(_ids), _ids)

        # concatenate the sampled tokens to the beam and prepare the
        # model outputs for the next layer; also compute if we
        # has finished decoding by predicting the end token
        ids = tf.concat([ids, _ids], 1)
        log_probs = log_probs + _log_probs[..., 0]
        return ([queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels,
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes],
                tf.logical_or(closed, tf.equal(_ids[:, 0], 3)))

    def beam_search(self,
                    inputs,
                    closed,
                    last_beam_size,
                    beam_size,
                    **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: Dataclass
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

        # compute a distribution over tokens
        logits = tf.math.log_softmax(self.logits_before_softmax(
                                     queries, **kwargs)[:, -1])
        batch_size = tf.shape(logits)[0] // last_beam_size

        # sample the top beam_size candidates
        _log_probs, _ids = tf.math.top_k(logits, k=beam_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(_log_probs)[:1], 0), beam_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, closed_log_probs, _log_probs)
        _ids = tf.where(mask, tf.zeros_like(_ids), _ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size, beam_size])
        _log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, 1]) + _log_probs
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size * beam_size])

        # select the top beam_size candidates
        _log_probs, beam_ids = tf.math.top_k(_log_probs, k=beam_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        old_beam_ids = tf.math.floordiv(beam_ids, beam_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        _ids = tf.reshape(_ids, [batch_size, last_beam_size * beam_size])
        _ids = tf.gather(_ids, beam_ids, batch_dims=1)
        _ids = tf.reshape(_ids, [batch_size * beam_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            if x is None:
                return x
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * beam_size], shape], axis=0)
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
        closed = select(closed)

        # TODO: Brandon -> handle the image features as well.
        object_detections = select(object_detections)
        object_features = select(object_features)
        object_boxes = select(object_boxes)

        # concatenate the sampled tokens to the beam and prepare the
        # model outputs for the next layer; also compute if we
        # has finished decoding by predicting the end token
        ids = tf.concat([ids, _ids], 1)
        log_probs = tf.reshape(_log_probs, [batch_size * beam_size])
        return ([queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels, 
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes],
                tf.logical_or(closed, tf.equal(_ids[:, 0], 3)), beam_size)

    def adaptive_search(self,
                        inputs,
                        closed,
                        last_beam_size,
                        beam_size,
                        natural_order_tokens,
                        natural_order_pos,
                        **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: Dataclass
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
        natural_order_tokens: tf.Tensor
            a batch of sequences representing the generation index of tokens
            in natural order that are yet to be decoded.
        natural_order_pos: tf.Tensor
            a batch of sequences representing the word ids of tokens
            in natural order that are yet to be decoded.

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model
        natural_order: tf.Tensor
            a batch of sequences representing the words in natural order
            that are yet to be decoded."""

        # unpack all the requires model inputs, some might be empty tensors:
        [queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels,
         partial_pos, pointer_probs, log_probs,
         object_detections, object_features, object_boxes] = inputs

        # compute a distribution over tokens
        logits = self.logits_before_softmax(queries, **kwargs)[:, -1]

        # calculate a mask over the vocab
        mask = tf.reduce_sum(tf.one_hot(
            natural_order_tokens, tf.shape(logits)[1], axis=2), axis=1)

        # make sure the mask is clipped to be 0.0 or 1.0
        mask = tf.clip_by_value(mask, 0.0, 1.0)

        # true if the mask contains <unk> or any word
        is_token = tf.logical_or(tf.equal(
            mask[:, 1], 1), tf.reduce_any(tf.equal(mask[:, 4:], 1), axis=1))

        # true if the mask does not contain any words, and contains <end>
        is_end = tf.logical_and(
            tf.logical_not(is_token), tf.equal(mask[:, 3], 1))

        # a mask that contains only the <end> token
        end_mask = tf.one_hot(
            tf.fill([tf.shape(logits)[0]], 3), tf.shape(logits)[1], axis=1)

        # a mask that contains only the <pad> token
        pad_mask = tf.one_hot(
            tf.fill([tf.shape(logits)[0]], 0), tf.shape(logits)[1], axis=1)

        # a mask that contains only words
        token_mask = tf.clip_by_value(mask - end_mask - pad_mask, 0.0, 1.0)

        # create a batch of different masks
        mask = tf.where(is_end[:, tf.newaxis],
                        end_mask, tf.where(is_token[:, tf.newaxis],
                                           token_mask, pad_mask))

        # convert the masks into offsets for the softmax op: 0 -> -\infty
        offset = (1.0 - mask) * 999999.0
        logits = tf.math.log_softmax(logits - offset)
        batch_size = tf.shape(logits)[0] // last_beam_size

        # sample the top beam_size candidates
        _log_probs, _ids = tf.math.top_k(logits, k=beam_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(_log_probs)[:1], 0), beam_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        _log_probs = tf.where(mask, closed_log_probs, _log_probs)
        _ids = tf.where(mask, tf.zeros_like(_ids), _ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size, beam_size])
        _log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, 1]) + _log_probs
        _log_probs = tf.reshape(_log_probs, [
            batch_size, last_beam_size * beam_size])

        # select the top beam_size candidates
        _log_probs, beam_ids = tf.math.top_k(_log_probs, k=beam_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        old_beam_ids = tf.math.floordiv(beam_ids, beam_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        _ids = tf.reshape(_ids, [batch_size, last_beam_size * beam_size])
        _ids = tf.gather(_ids, beam_ids, batch_dims=1)
        _ids = tf.reshape(_ids, [batch_size * beam_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            if x is None:
                return x
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * beam_size], shape], axis=0)
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
        closed = select(closed)
        natural_order_tokens = select(natural_order_tokens)
        natural_order_pos = select(natural_order_pos)

        # TODO: Brandon -> handle the image features as well.
        object_detections = select(object_detections)
        object_features = select(object_features)
        object_boxes = select(object_boxes)

        # concatenate the sampled tokens to the beam and prepare the
        # model outputs for the next layer; also compute if we
        # has finished decoding by predicting the end token
        ids = tf.concat([ids, _ids], 1)
        log_probs = tf.reshape(_log_probs, [batch_size * beam_size])
        return ([queries, values, queries_mask, values_mask, ids, permutation,
                absolute_positions, relative_positions,
                pointer_labels, logits_labels,
                partial_pos, pointer_probs, log_probs,
                object_detections, object_features, object_boxes],
                tf.logical_or(closed, tf.equal(_ids[:, 0], 3)),
                beam_size, natural_order_tokens, natural_order_pos)

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(output_size=self.output_size,
                      logits_embedding=self.logits_embedding,
                      label_smoothing=self.label_smoothing,
                      ** self.kwargs)

        base_config = super(Logits, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

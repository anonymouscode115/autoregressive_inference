from voi.data.load import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.nn.input import TransformerInput
from voi.nn.input import RegionFeatureInput
from voi.algorithms.adaptive_search import adaptive_search
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.permutation_utils import get_permutation
from voi.birkoff_utils import birkhoff_von_neumann
from voi.data.tagger import load_parts_of_speech
from voi.data.tagger import load_tagger
from scipy import stats
import tensorflow as tf
import os
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


def levenshtein(seq1, seq2):
    # https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1)
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1)
    return matrix[size_x - 1, size_y - 1]


np.set_printoptions(threshold=np.inf)

coco_batch_spec = [{
    'image_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'image_path': tf.TensorSpec(shape=[None], dtype=tf.string),
    'tags': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'global_features': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'scores': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'boxes': tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    'labels': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'boxes_features': tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)}]

wmt_batch_spec = [{
    'encoder_words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'encoder_token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'decoder_words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'decoder_token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32)}]


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)]
                             + coco_batch_spec)
def prepare_batch_for_lm_captioning(action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to refine
        such that we choose the one with the lowest loss
        see https://arxiv.org/pdf/1512.07679.pdf
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    def repeat_tensor_list(lst, n):
        for i in range(len(lst)):
            if isinstance(lst[i], tf.Tensor):
                lst[i] = tf.repeat(lst[i], n, axis=0)
        return lst

    # select all relevant features from the batch dictionary
    image_ind = batch["image_indicators"]
    boxes_features = batch["boxes_features"]
    boxes = batch["boxes"]
    detections = batch["labels"]
    words = batch["words"]
    mask = batch["token_indicators"]
    batch_size = tf.shape(mask)[0]
    return repeat_tensor_list([words[:, :-1], tf.zeros([batch_size]),
                               tf.greater(mask[:, :-1], 0), tf.greater(image_ind, 0),
                               words[:, 1:], None, None, None, None, None, None, tf.zeros([batch_size]),
                               tf.zeros([batch_size]), detections, boxes_features, boxes], action_refinement)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)]
                             + wmt_batch_spec)
def prepare_batch_for_lm_wmt(action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to refine
        such that we choose the one with the lowest loss
        see https://arxiv.org/pdf/1512.07679.pdf
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    def repeat_tensor_list(lst, n):
        for i in range(len(lst)):
            if isinstance(lst[i], tf.Tensor):
                lst[i] = tf.repeat(lst[i], n, axis=0)
        return lst

    # select all relevant features from the batch dictionary
    encoder_words = batch["encoder_words"]
    encoder_token_ind = batch["encoder_token_indicators"]
    words = batch["decoder_words"]
    mask = batch["decoder_token_indicators"]
    batch_size = tf.shape(mask)[0]

    return repeat_tensor_list([words[:, :-1], encoder_words,
                               tf.greater(mask[:, :-1], 0), tf.greater(encoder_token_ind, 0),
                               words[:, 1:], None, None, None, None, None, None, tf.zeros([batch_size]),
                               tf.zeros([batch_size]), tf.zeros([batch_size]),
                               tf.zeros([batch_size]), tf.zeros([batch_size, 1])], action_refinement)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
                              tf.TensorSpec(shape=None, dtype=tf.int32)]
                             + coco_batch_spec)
def prepare_batch_for_pt_captioning(pretrain_done, action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    pretrain_done: tf.bool
        whether decoder pretraining has done
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to refine
        such that we choose the one with the lowest loss
        see https://arxiv.org/pdf/1512.07679.pdf
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # select all relevant features from the batch dictionary
    image_ind = batch["image_indicators"]
    boxes_features = batch["boxes_features"]
    boxes = batch["boxes"]
    detections = batch["labels"]
    words = batch["words"]
    batch_size = tf.shape(words)[0]

    start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

    l2r_relative = pt_permutation_to_relative_l2r(tf.shape(words)[0],
                                                  tf.shape(words)[1],
                                                  tf.constant(10))

    return [words, None,
            tf.logical_not(start_end_or_pad), tf.greater(image_ind, 0),
            pretrain_done, action_refinement,
            None, l2r_relative, None, None, None, None, None,
            detections, boxes_features, boxes]


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
                              tf.TensorSpec(shape=None, dtype=tf.int32)]
                             + wmt_batch_spec)
def prepare_batch_for_pt_wmt(pretrain_done, action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    pretrain_done: tf.bool
        whether decoder pretraining has done
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to refine
        such that we choose the one with the lowest loss
        see https://arxiv.org/pdf/1512.07679.pdf
    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # select all relevant features from the batch dictionary
    encoder_words = batch["encoder_words"]
    encoder_token_ind = batch["encoder_token_indicators"]
    words = batch["decoder_words"]
    mask = batch["decoder_token_indicators"]
    batch_size = tf.shape(words)[0]

    start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

    l2r_relative = pt_permutation_to_relative_l2r(tf.shape(words)[0],
                                                  tf.shape(words)[1],
                                                  tf.constant(10))

    return [words, encoder_words,
            tf.logical_not(start_end_or_pad), tf.greater(encoder_token_ind, 0),
            pretrain_done, action_refinement,
            None, l2r_relative, None, None, None, None, None,
            None, None, None]


def prepare_permutation(batch,
                        tgt_vocab_size,
                        order,
                        dataset,
                        policy_gradient,
                        decoder=None):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    tgt_vocab_size: tf.Tensor
        the number of words in the target vocabulary of the model; used in order
        to calculate labels for the language model logits
    order: str or callable
        the autoregressive ordering to train Transformer-InDIGO using;
        l2r or r2l for now, will support soft orders later
    dataset: str
        type of dataset (captioning or wmt)
    policy_gradient:
        whether to use policy gradient for training
        choices:
            none: (no policy gradient)
            with_bvn: use policy gradient with probabilities of
                hard permutations based on Berkhoff von Neumann decomposition
                of soft permutation
            without_bvn: after applying Hungarian algorithm on soft
                permutation to obtain hard permutations, the probabilities of hard
                permutations are proportionally based on Gumbel-Matching distribution
                i.e. exp(<X,P>_F), see https://arxiv.org/abs/1802.08665)

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # process the dataset batch dictionary into the standard
    # model input format

    if dataset == 'captioning':
        words = batch['words']
        mask = batch['token_indicators']
        prepare_batch_for_lm = prepare_batch_for_lm_captioning
        prepare_batch_for_pt = prepare_batch_for_pt_captioning
    elif dataset in ['wmt', 'django', 'gigaword']:
        words = batch['decoder_words']
        mask = batch['decoder_token_indicators']
        prepare_batch_for_lm = prepare_batch_for_lm_wmt
        prepare_batch_for_pt = prepare_batch_for_pt_wmt

    inputs = prepare_batch_for_lm(tf.constant(1), batch)
    permu_inputs = None
    # the order is fixed
    if order in ['r2l', 'l2r', 'rare', 'common', 'test']:
        inputs[5] = get_permutation(mask, words, tf.constant(order))

    # pass the training example through the permutation transformer
    # to obtain a doubly stochastic matrix
    if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
        if policy_gradient != 'without_bvn':
            inputs[5] = order(prepare_batch_for_pt(tf.constant(True),
                                                   tf.constant(1), batch), training=True)
        else:
            permu_inputs = prepare_batch_for_pt(tf.constant(True),
                                                tf.constant(1), batch)
            inputs[5], activations, kl, log_nom, log_denom = \
                order(permu_inputs, training=True)
            permu_inputs[-6] = activations
            permu_inputs[-5] = kl
            permu_inputs[-4] = log_nom - log_denom

    # pass the training example through the permutation transformer
    # to obtain a doubly stochastic matrix
    if order == 'sao' and decoder is not None:
        cap, logp, rel_pos = adaptive_search(
            inputs, decoder, dataset,
            beam_size=8, max_iterations=200, return_rel_pos=True)
        pos = tf.argmax(rel_pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(pos), axis=2)
        pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.float32)
        ind = tf.random.uniform([tf.shape(pos)[0], 1], maxval=7, dtype=tf.int32)
        # todo: make sure this is not transposed
        inputs[5] = tf.squeeze(tf.gather(pos, ind, batch_dims=1), 1)

    if policy_gradient == 'with_bvn':
        raise NotImplementedError
    elif policy_gradient == 'without_bvn':
        inputs[5] = tf.stop_gradient(inputs[5])

    # convert the permutation to absolute and relative positions
    inputs[6] = inputs[5][:, :-1, :-1]
    inputs[7] = permutation_to_relative(inputs[5])

    # convert the permutation to label distributions
    # also records the partial absolute position at each decoding time step
    hard_pointer_labels, inputs[10] = permutation_to_pointer(inputs[5][:, tf.newaxis, :, :])
    inputs[8] = tf.squeeze(hard_pointer_labels, axis=1)
    inputs[9] = tf.matmul(inputs[5][
                          :, 1:, 1:], tf.one_hot(inputs[4], tf.cast(tgt_vocab_size, tf.int32)))

    return inputs, permu_inputs


def inspect_time_dataset(tfrecord_folder,
                         ref_folder,
                         batch_size,
                         beam_size,
                         model,
                         model_ckpt,
                         order,
                         vocabs,
                         strategy,
                         policy_gradient,
                         save_path,
                         dataset_type,
                         tagger_file):
    """
    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    ref_folder: str
        the path to a folder that contains ground truth sentence files
        ready to be loaded from the disk
    batch_size: int
        the maximum number of training examples in a
        single batch
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch
    model: Decoder
        the caption model to be validated; an instance of Transformer that
        returns a data class TransformerInput
    model_ckpt: str
        the path to an existing model checkpoint or the path
        to be written to when training
    order: tf.keras.Model
        the autoregressive ordering to train Transformer-InDIGO using;
        must be a keras model that returns permutations
    vocabs: list of Vocabulary
        the model vocabulary which contains mappings
        from words to integers
    strategy: tf.distribute.Strategy
        the strategy to use when distributing a model across many gpus
        typically a Mirrored Strategy
    policy_gradient: str
        whether to use policy gradient for training
        default: none (no policy gradient)
        choices:
            with_bvn: use policy gradient with probabilities of
                hard permutations based on Berkhoff von Neumann decomposition
                of soft permutation
            without_bvn: after applying Hungarian algorithm on soft
                permutation to obtain hard permutations, the probabilities of hard
                permutations are proportionally based on Gumbel-Matching distribution
                i.e. exp(<X,P>_F), see https://arxiv.org/abs/1802.08665)
    save_path: str
        save path for parts of speech analysis
    dataset_type: str
        the type of dataset"""

    def pretty(s):
        return s.replace('_', ' ').title()

    tagger = load_tagger(tagger_file)
    tagger_vocab = load_parts_of_speech()

    # create a validation pipeline
    if dataset_type == 'captioning':
        dataset = faster_rcnn_dataset(tfrecord_folder, batch_size, shuffle=False)
        prepare_batch_for_lm = prepare_batch_for_lm_captioning
        prepare_batch_for_pt = prepare_batch_for_pt_captioning
    elif dataset_type in ['wmt', 'django', 'gigaword']:
        dataset = wmt_dataset(tfrecord_folder, batch_size, shuffle=False)
        prepare_batch_for_lm = prepare_batch_for_lm_wmt
        prepare_batch_for_pt = prepare_batch_for_pt_wmt
    dataset = strategy.experimental_distribute_dataset(dataset)

    def dummy_loss_function(b):
        # process the dataset batch dictionary into the standard
        # model input format
        inputs, permu_inputs = prepare_permutation(
            b, vocabs[-1].size(),
            order, dataset_type, policy_gradient, decoder=model)
        _ = model(inputs)
        loss, inputs = model.loss(inputs, training=True)
        permu_loss = tf.zeros(tf.shape(loss)[0])

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_dummy_loss_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        strategy.run(dummy_loss_function, args=(b,))

    # run the model for a single forward pass
    # and load en existing checkpoint into the trained model
    for batch in dataset:
        wrapped_dummy_loss_function(batch)
        break

    print("----------Done defining weights of model-----------")

    if tf.io.gfile.exists(model_ckpt):
        model.load_weights(model_ckpt)
    if tf.io.gfile.exists(model_ckpt.replace(".", ".pt.")):
        order.load_weights(model_ckpt.replace(".", ".pt."))

    # for captioning
    ref_caps = {}
    hyp_caps = {}
    gen_order_caps = {}
    # for non-captioning
    ref_caps_list = []
    hyp_caps_list = []
    gen_order_list = []

    order_words_raw = np.ones(vocabs[-1].size(), dtype=np.float32) * (-1e-4)
    num_words_raw = np.ones(vocabs[-1].size(), dtype=np.float32) * 1e-4

    # create data frames for global sequence-level statistics
    time_stats_df = pd.DataFrame(columns=[
        'Model',
        'Type',
        'Sequence Length',
        'Time'])

    def decode_function(b):
        # calculate the ground truth sequence for this batch; and
        # perform beam search using the current model
        # show several model predicted sequences and their likelihoods
        inputs = prepare_batch_for_lm(tf.constant(1), b)
        start_time_sao = tf.timestamp()
        with tf.control_dependencies([start_time_sao]):
            cap, logp, rel_pos = adaptive_search(
                inputs, model, dataset_type, beam_size=beam_size, max_iterations=50,
                return_rel_pos=True)
        with tf.control_dependencies([cap, logp, rel_pos]):
            stop_time_sao = tf.timestamp()
        permu = None
        start_time_pt = 0
        stop_time_pt = 0
        if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
            if policy_gradient != 'without_bvn':
                permu_inputs = prepare_batch_for_pt(tf.constant(True), tf.constant(1), b)
                start_time_pt = tf.timestamp()
                with tf.control_dependencies([start_time_pt]):
                    permu = order(permu_inputs)
                with tf.control_dependencies([permu]):
                    stop_time_pt = tf.timestamp()
            else:
                permu_inputs = prepare_batch_for_pt(tf.constant(True), tf.constant(1), b)
                start_time_pt = tf.timestamp()
                with tf.control_dependencies([start_time_pt]):
                    permu, _, _, _, _ = order(permu_inputs)
                with tf.control_dependencies([permu]):
                    stop_time_pt = tf.timestamp()
        return cap, logp, rel_pos, permu, \
               stop_time_sao - start_time_sao, \
               stop_time_pt - start_time_pt

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_decode_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        return strategy.run(decode_function, args=(b,))

    # loop through the entire dataset once (one epoch)
    if dataset_type in ['wmt', 'django', 'gigaword']:
        f = open(save_path, "w")

    for b_num, batch in enumerate(dataset):

        if dataset_type in ['wmt', 'django', 'gigaword']:
            bw = batch["decoder_words"]
        elif dataset_type == 'captioning':
            bw = batch["words"]
        if strategy.num_replicas_in_sync == 1:
            batch_wordids = bw
        else:
            batch_wordids = tf.concat(bw.values, axis=0)

        if dataset_type == 'captioning':
            if strategy.num_replicas_in_sync == 1:
                paths = [x.decode("utf-8") for x in batch["image_path"].numpy()]
            else:
                paths = [x.decode("utf-8") for x in tf.concat(batch["image_path"].values, axis=0).numpy()]
            paths = [os.path.join(ref_folder, os.path.basename(x)[:-7] + "txt")
                     for x in paths]

            # iterate through every ground truth training example and
            # select each row from the text file
            for file_path in paths:
                with tf.io.gfile.GFile(file_path, "r") as f:
                    ref_caps[file_path] = [
                        x for x in f.read().strip().lower().split("\n")
                        if len(x) > 0]

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        cap, log_p, rel_pos, permu, \
        sao_time, pt_time = wrapped_decode_function(batch)
        if strategy.num_replicas_in_sync == 1:
            caparr, logparr, relposarr, permuarr = [cap], [log_p], [rel_pos], [permu]
            sao_timearr, pt_timeparr = [sao_time], [pt_time]
        else:
            caparr, logparr, relposarr, permuarr = \
                cap.values, log_p.values, rel_pos.values, permu.values
            sao_timearr, pt_timeparr = \
                sao_time.values, pt_time.values
        #             cap = tf.concat(cap.values, axis=0)
        #             log_p = tf.concat(log_p.values, axis=0)
        #             rel_pos = tf.concat(rel_pos.values, axis=0)
        for nzip, tmp in enumerate(zip(caparr, logparr, relposarr, permuarr,
                                       sao_timearr, pt_timeparr)):
            cap, log_p, rel_pos, permu, sao_time, pt_time = tmp
            # get the absolute position because the output of decoder
            # is a list of words whose order is determined by the
            # relative position matrix
            pos = tf.argmax(rel_pos, axis=-1, output_type=tf.int32) - 1
            pos = tf.reduce_sum(tf.nn.relu(pos[:, :, 1:, 1:]), axis=2)
            pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.int32)

            # calculate the generation order of captions
            gen_order_cap = tf.squeeze(tf.matmul(pos, cap[..., tf.newaxis]), axis=-1)

            cap_id = cap

            # generate a mask over valid words
            mask = tf.cast(tf.math.logical_not(
                tf.math.equal(cap_id, 0)), tf.float32)

            cap = tf.strings.reduce_join(
                vocabs[-1].ids_to_words(cap), axis=2, separator=' ').numpy()
            gen_order_cap = tf.strings.reduce_join(
                vocabs[-1].ids_to_words(gen_order_cap), axis=2, separator=' ').numpy()

            # format the model predictions into a string; the evaluation package
            # requires input to be strings; not there will be slight
            # formatting differences between ref and hyp
            for i in range(cap.shape[0]):
                real_i = nzip * cap.shape[0] + i
                if dataset_type == 'captioning' and paths[real_i] not in hyp_caps:
                    print(real_i)
                    hyp_caps[paths[real_i]] = cap[i, 0].decode("utf-8")
                    gen_order_caps[paths[real_i]] = gen_order_cap[i, 0].decode("utf-8")

                    if isinstance(order, tf.keras.Model):
                        print(b_num, "PT Permutation:\n", permu[i].numpy())
                        print(b_num, "Ground truth: {} | PT: {}".format(
                            tf.strings.reduce_join(
                                vocabs[-1].ids_to_words(batch_wordids[real_i]),
                                separator=' ').numpy(),
                            tf.strings.reduce_join(
                                vocabs[-1].ids_to_words(tf.squeeze(
                                    tf.matmul(tf.cast(permu[i], tf.int32),
                                              batch_wordids[real_i][:, tf.newaxis]))),
                                separator=' ').numpy()))

                    for j in range(log_p.shape[1]):

                        print("{}: [p = {}] {} | {}".format(
                            paths[i],
                            np.exp(log_p[i, j].numpy()),
                            cap[i, j].decode("utf-8"),
                            gen_order_cap[i, j].decode("utf-8")))

                        print("Decoder Permutation:\n", pos[i, j].numpy())

                        # the length of the sentence as an independent variable
                        seq_len = int(mask[i, j].numpy().sum()) - 1  # get rid of the end token

                        time_stats_df = time_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "SAO",
                            'Sequence Length': seq_len,
                            'Time': float(sao_time.numpy().sum())},
                            ignore_index=True)
                        time_stats_df = time_stats_df.append({
                            "Model": model_ckpt,
                            "Type": "VOI",
                            'Sequence Length': seq_len,
                            'Time': float(pt_time.numpy().sum())},
                            ignore_index=True)

                elif dataset_type != 'captioning':
                    if "<unk>" not in tf.strings.reduce_join(
                            vocabs[-1].ids_to_words(batch_wordids[real_i]),
                            separator=' ').numpy().decode("utf-8"):
                        hyp_caps_list.append(cap[i, 0].decode("utf-8"))
                        gen_order_list.append(gen_order_cap[i, 0].decode("utf-8"))

                        if isinstance(order, tf.keras.Model):
                            print("PT Permutation:\n", permu[i].numpy(), file=f)
                            print("Ground truth: {} | PT: {}".format(
                                tf.strings.reduce_join(
                                    vocabs[-1].ids_to_words(batch_wordids[real_i]),
                                    separator=' ').numpy(),
                                tf.strings.reduce_join(
                                    vocabs[-1].ids_to_words(tf.squeeze(
                                        tf.matmul(tf.cast(permu[i], tf.int32),
                                                  batch_wordids[real_i][:, tf.newaxis]))),
                                    separator=' ').numpy()), file=f)

                        for j in range(log_p.shape[1]):
                            print("[p = {}] {} | {}".format(np.exp(log_p[i, j].numpy()),
                                                            cap[i, j].decode("utf-8"),
                                                            gen_order_cap[i, j].decode("utf-8")), file=f)
                            print("Decoder Permutation:\n", pos[i, j].numpy(), file=f)

        # process the logged metrics about order
        time_stats_df.to_csv(f'{model_ckpt}_time_stats_df.csv')

        plt.clf()
        g = sns.relplot(x='Sequence Length',
                        y='Time',
                        hue='Type',
                        data=time_stats_df,
                        kind="line",
                        height=5,
                        aspect=2,
                        facet_kws={"legend_out": True})
        g.set(title='Search Times For SAO vs VOI')
        plt.savefig(f'{model_ckpt}_timing.png',
                    bbox_inches='tight')
        plt.close()

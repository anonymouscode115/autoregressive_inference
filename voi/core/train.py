from voi.core.sess_util import get_session
from voi.data.load import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.algorithms.beam_search import beam_search
from voi.algorithms.adaptive_search import adaptive_search
from voi.birkoff_utils import birkhoff_von_neumann
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import pt_permutation_to_relative_l2r
from voi.permutation_utils import get_permutation
from voi.running_mean_std import RunningMeanStd
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import pandas as pd

"""inputs = (queries, values, queries_mask, values_mask, ids, permutation,
         absolute_positions, relative_positions,
         pointer_labels, logits_labels, partial_pos, 
         pointer_probs, log_probs,
         object_detections, object_features, object_boxes)
   permu_inputs = (queries, values, queries_mask, values_mask, pretrain_done, action_refinement,
         absolute_pos, rel_pos, None, None, activations, kl, noise_for_log_prob,
         object_detections, object_features, object_boxes)    
   activations, kl, noise_for_log_prob are returns from policy gradient
   with Gumbel Sinkhorn (without BvN)
"""

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
    
# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)]
#                              + coco_batch_spec)
def prepare_batch_for_lm_captioning(action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:
    
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data
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

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)]
#                              + wmt_batch_spec)
def prepare_batch_for_lm_wmt(action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:
    
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data
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

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
#                              tf.TensorSpec(shape=None, dtype=tf.int32)] 
#                              + coco_batch_spec)
def prepare_batch_for_pt_captioning(pretrain_done, action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    pretrain_done: tf.bool
        whether decoder pretraining has done
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data     
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
                                                  tf.constant(10)) # unused unless we use DecoderWithPositionLayer for PT
    
    return [words, None,
            tf.logical_not(start_end_or_pad), tf.greater(image_ind, 0),
            pretrain_done, action_refinement, 
            None, l2r_relative, None, None, None, None, None,
            detections, boxes_features, boxes]

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
#                              tf.TensorSpec(shape=None, dtype=tf.int32)] 
#                              + wmt_batch_spec)
def prepare_batch_for_pt_wmt(pretrain_done, action_refinement, batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    pretrain_done: tf.bool
        whether decoder pretraining has done
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data       
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
                                                  tf.constant(10)) # unused unless we use DecoderWithPositionLayer for PT
    
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
                        pretrain_done,
                        action_refinement,
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
    pretrain_done: tf.bool
        for policy gradient, whether decoder pretraining has finished
    action_refinement: tf.int32
        in policy gradient, the number of actions (permutations) to sample
        per training data  

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
        
    inputs = prepare_batch_for_lm(action_refinement, batch)
    permu_inputs = None
    # the order is fixed
    if order in ['r2l', 'l2r', 'rare', 'common', 'test']:
        inputs[5] = get_permutation(mask, words, tf.constant(order))

    # pass the training example through the permutation transformer
    # to obtain a doubly stochastic matrix
    if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
        if policy_gradient != 'without_bvn':
            inputs[5] = order(prepare_batch_for_pt(pretrain_done,
                                  action_refinement, batch), training=True)
        else:
            permu_inputs = prepare_batch_for_pt(pretrain_done,
                               action_refinement, batch)
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
        
    # apply the birkhoff-von neumann decomposition to support general
    # doubly stochastic matrices
    #p, c = birkhoff_von_neumann(inputs[5], tf.constant(20))
    #c = c / tf.reduce_sum(c, axis=1, keepdims=True)

    # convert the permutation to absolute and relative positions
    inputs[6] = inputs[5][:, :-1, :-1]
    inputs[7] = permutation_to_relative(inputs[5])

    # convert the permutation to label distributions
    # also records the partial absolute position at each decoding time step
    hard_pointer_labels, inputs[10] = permutation_to_pointer(inputs[5][:, tf.newaxis, :, :])
    inputs[8] = tf.squeeze(hard_pointer_labels, axis=1)
    inputs[9] = tf.matmul(inputs[5][
        :, 1:, 1:], tf.one_hot(inputs[4], tf.cast(tgt_vocab_size, tf.int32)))
    
#     inputs[7] = tf.reduce_sum(permutation_to_relative(
#         p) * c[..., tf.newaxis, tf.newaxis, tf.newaxis], axis=1)

#     # convert the permutation to label distributions
#     # also records the partial absolute position at each decoding time step
#     hard_pointer_labels, inputs[10] = permutation_to_pointer(p)
#     inputs[8] = tf.reduce_sum(
#         hard_pointer_labels * c[..., tf.newaxis, tf.newaxis], axis=1)
#     inputs[9] = tf.matmul(inputs[5][
#         :, 1:, 1:], tf.one_hot(inputs[4], tf.cast(tgt_vocab_size, tf.int32)))

    return inputs, permu_inputs


def train_dataset(train_folder,
                  batch_size,
                  beam_size,
                  num_epoch,
                  model,
                  model_ckpt,
                  save_interval,
                  order,
                  vocabs,
                  strategy,
                  dataset,
                  policy_gradient,
                  reward_std,
                  pg_final_layer,
                  decoder_pretrain,
                  decoder_init_lr,
                  pt_init_lr,
                  lr_schedule,
                  warmup,
                  kl_coeff,
                  kl_log_linear,
                  action_refinement,
                  alternate_training,
                  use_ppo,
                  embedding_align_coeff,
                  decoder_training_scheme):
    """Trains a transformer based caption model using features extracted
    using a facter rcnn object detection model

    Arguments:

    train_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk;
        used for training
    batch_size: int
        the maximum number of training examples in a
        single batch
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch for visualization
    num_epochs: int
        the number of loops through the entire dataset to
        make before termination
    model: Decoder i.e. tf.keras.Model
        the decoder model to be trained that generates 
        target sequence through insertion; implemented as Transformer-INDIGO
    model_ckpt: str
        the path to an existing model checkpoint or the path
        to be written to when training
    save_interval: int
        the model snapshot saving interval;
        the snapshot is named by the # of iterations; 
        disable if <=0; 
        (model_name_ckpt.h5 etc still saved every 2000 iterations)
    order: str / tf.keras.Model
        the type of autoregressive ordering to train;
        either fixed ordering (e.g. L2R) specified through a string,
        or VOI, in which case the order is tf.keras.Model
    vocabs: list of Vocabulary
        the list of model vocabularies which contains mappings
        from words to integers
    strategy: tf.distribute.Strategy
        the strategy to use when distributing a model across many gpus
        typically a Mirrored Strategy
    dataset: str
        the type of dataset (captioning, django, gigaword, or wmt)
    policy_gradient: str
        whether to use policy gradient for training;
        this parameter is used for VOI training only
        default: none (no policy gradient, which is necessary for
                       fixed ordering training)
        choices: 
            with_bvn: use policy gradient with probabilities of 
                hard permutations based on Berkhoff von Neumann decomposition
                of doubly stochastic matrix (IN PRACTICE THIS HAS NUMERICAL ISSUES SO 
                WE HAVE DISABLED IT)
            without_bvn: after applying the Hungarian algorithm on soft 
                permutation from the Sinkhorn operation to obtain hard permutations, 
                i.e. P = Hungarian(Sinkhorn((X + eps) / self.temperature)),
                where X = q(y, x),
                the probabilities of hard 
                permutations are proportionally based on Gumbel-Matching distribution 
                i.e. exp(<X,P>_F), see https://arxiv.org/abs/1802.08665) 
    reward_std: bool
        for policy gradient, whether to standardize the reward;
        this parameter is used for VOI training only
    pg_final_layer: str
        the type of final layer for permutation transformer if training VOI; 
        default is sinkhorn;
        this parameter is used for VOI training only    
    decoder_pretrain: int
        decoder pretraining timesteps before start training the permutation transformer;
        this parameter is used for VOI training only
    decoder_init_lr: float
        decoder transformer's learning rate at initialization
    pt_init_lr: float
        permutation transformer's learning rate at initialization
    lr_schedule: str
        learning rate schedule: linear or constant
    warmup: int
        number of timesteps for linearly warming up the learning rate       
    kl_coeff: float
        kl divergence coefficient beta
        where the loss is beta * KL((X+eps)/self.temperature || eps/temp_prior),
        where eps is Gumbel noise;
        this parameter is used for VOI training only
    kl_log_linear: float
        if this value > 0, decrease the log coefficient of kl
        linearly as training proceeds until this value;
        this parameter is used for VOI training only
    action_refinement: int
        the number of actions (permutations, orderings) to sample
        per training data;
        this parameter is used for VOI training only        
    alternate_training: list of two ints or None
        if alternate training is not None, 
        then train decoder and fix permutation transformer for x iterations, 
        and then train permutation transformer and fix decoder for y iterations
    use_ppo: bool
        whether to use PPO;
        this parameter is used for VOI training only
    embedding_align_coeff: float
        the coefficient of embedding alignment loss 
        between the permutation transformer and the decoder;
        this parameter is used for VOI training only    
    decoder_training_scheme: str
        whether to train decoder with the best permutation 
        or all sampled permutations from the permutation transformer;
        this parameter is used for VOI training only""" 

    if dataset == 'captioning':
        train_dataset = faster_rcnn_dataset(train_folder, batch_size)
    elif dataset in ['wmt', 'django', 'gigaword']:
        train_dataset = wmt_dataset(train_folder, batch_size)
        
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # create an optimizer
    init_lr = decoder_init_lr
    pt_init_lr = pt_init_lr
    
    with strategy.scope():
        optim = tf.keras.optimizers.Adam(learning_rate=init_lr, 
                                         beta_2=0.98 if dataset in ['wmt', 'django', 'gigaword'] else 0.999)
        pt_optim = tf.keras.optimizers.Adam(learning_rate=pt_init_lr)    

    reward_normalizer = RunningMeanStd() # UNUNSED
    
    if dataset == 'captioning':
        prepare_batch_for_lm = prepare_batch_for_lm_captioning
        prepare_batch_for_pt = prepare_batch_for_pt_captioning
    elif dataset in ['wmt', 'django', 'gigaword']:
        prepare_batch_for_lm = prepare_batch_for_lm_wmt
        prepare_batch_for_pt = prepare_batch_for_pt_wmt   
    
    # Get the number of embeddings matrices in PT and decoder
    # In captioning, there is a detection embedding and vocab embedding
    # In other tasks, this equals the number of vocabularies
    def get_num_emb_matrices():
        if dataset == 'captioning':
            return 2
        elif dataset in ['wmt', 'django', 'gigaword']:
            return len(vocabs)
    
    num_emb_matrices = get_num_emb_matrices()
            
    def calc_align_loss():
        # embedding_weights = [Mirrored{0: (27000x512), 1: (27000x512), ....}]
        model_e = model.src_embedding.weights
        order_e = order.src_embedding.weights
        cos_src = tf.reduce_sum(tf.math.multiply(model_e, order_e), 
                    axis=-1) / (tf.linalg.norm(model_e, axis=-1)
                                * tf.linalg.norm(order_e, axis=-1))
        cos_src = tf.reduce_mean(cos_src)
        if num_emb_matrices == 1:
            return 2.0 * (1 - cos_src)
        elif num_emb_matrices == 2:
            model_e = model.tgt_embedding.weights
            order_e = order.tgt_embedding.weights
            cos_tgt = tf.reduce_sum(tf.math.multiply(model_e, order_e), 
                        axis=-1) / (tf.linalg.norm(model_e, axis=-1)
                                    * tf.linalg.norm(order_e, axis=-1)) 
            cos_tgt = tf.reduce_mean(cos_tgt)
            return (1 - cos_src) + (1 - cos_tgt)
                                     
            
    def loss_function(pretrain_done, kl_coeff, b):
        # process the dataset batch dictionary into the standard
        # model input format
        inputs, permu_inputs = prepare_permutation(
            b, vocabs[-1].size(),
            order, dataset, policy_gradient,
            pretrain_done, tf.constant(action_refinement), decoder=model)
        loss, inputs = model.loss(inputs, training=True)
        if dataset == 'captioning':
            token_ind = b['token_indicators']
        elif dataset in ['wmt', 'django', 'gigaword']:
            token_ind = b['decoder_token_indicators']
        repeated_tokeninds = tf.repeat(token_ind[:, 1:], action_refinement, axis=0)
        loss = tf.reduce_sum(loss * repeated_tokeninds, axis=1)  
        loss_with_refinement = loss
        
        # choose the best among action_refinement candidates
        reshaped_loss = tf.reshape(loss, [-1, action_refinement])
        best_idxs = tf.cast(tf.math.argmin(reshaped_loss, axis=1), tf.int32)
        best_idxs = tf.range(tf.shape(reshaped_loss)[0]) * action_refinement + best_idxs
        
        def mem_efficient_gather(t, idxs):
            partitions = tf.reduce_sum(tf.one_hot(idxs, tf.shape(t)[0], dtype='int32'), 0)
            # Selecting the elements we want to choose.
            return tf.dynamic_partition(t, partitions, 2)[1]
            
        def gather_tensor_lst(lst, idxs):
            for i in range(len(lst)):
                if isinstance(lst[i], tf.Tensor):
                    lst[i] = mem_efficient_gather(lst[i], idxs)
            return lst
        
        sampled_permus = inputs[5]
        # tf.dynamic_partition bug not resolved yet in tf 2.3
        # https://github.com/tensorflow/tensorflow/issues/42229        
        #inputs = gather_tensor_lst(inputs, best_idxs)
        #loss = mem_efficient_gather(loss, best_idxs)
        
        if policy_gradient != 'without_bvn':
            # without policy gradient or policy gradient + BvN
            # + stick breaking (unimplemented)
            permu_loss = tf.zeros(tf.shape(loss)[0])
            align_loss = tf.zeros(tf.shape(loss)[0])
        else:
            # sinkhorn permutation marginal inference

            #tf.print("reward:", -loss_with_refinement, summarize=-1)
            reward_baseline = tf.reduce_mean(-reshaped_loss, axis=1)
            std_baseline = tf.math.reduce_std(-reshaped_loss, axis=1) + 1e-8
            reward_baseline = tf.repeat(reward_baseline, action_refinement, axis=0)
            std_baseline = tf.repeat(std_baseline, action_refinement, axis=0)
            #tf.print("reward_baseline:", reward_baseline, summarize=-1)
            if reward_std:
                advantage = (tf.stop_gradient(-loss_with_refinement) - tf.stop_gradient(reward_baseline)) / \
                                tf.maximum(tf.stop_gradient(std_baseline), 1.0)
            else:
                advantage = tf.stop_gradient(-loss_with_refinement) - tf.stop_gradient(reward_baseline)
            #kl_factor = 1.0 #tf.math.reduce_std(advantage) + 1e-8
            advantage = tf.reshape(advantage, [-1, 1])
    
            log_potential = permu_inputs[-4]        
            for idx in range(3):
                locs = tf.where(inputs[5][idx] == 1.0)
                d2 = tf.shape(locs)[1]
                locs = tf.reshape(locs, [locs[-1,0]+1, d2])
                tf.print("Sampled 3 permutations:",
                        locs[:, -1], "\n", summarize=-1)

            log_potentials = log_potential
            advantages = advantage
            kls = permu_inputs[-5]
#             kls = tf.math.maximum(kls - 10.0, 0.0)
#             kls = kls ** 2
            #tf.print("log_potentials", tf.squeeze(log_potentials), summarize=-1)
            #tf.print("advantages", tf.squeeze(advantages), summarize=-1)
            permu_loss = -tf.reduce_mean(log_potentials * advantages - kl_coeff * kls, axis=1)
            align_loss = calc_align_loss()
            #tf.print("align loss", align_loss, summarize=-1)
            align_loss = embedding_align_coeff * align_loss * tf.ones_like(permu_loss)
                
        
        if decoder_training_scheme == 'best':
            # tf.dynamic_partition bug not resolved yet in tf 2.3
            # https://github.com/tensorflow/tensorflow/issues/42229
            raise NotImplementedError
            loss = loss / tf.reduce_sum(token_ind[:, 1:], axis=1)   
        else:
            loss = loss_with_refinement / tf.reduce_sum(repeated_tokeninds[:, 1:], axis=1)   
        
        bs = batch_size
        if decoder_training_scheme == 'all':
            bs = bs * action_refinement
            
        if policy_gradient != 'without_bvn':
            return tf.nn.compute_average_loss(loss, global_batch_size=bs), permu_loss, align_loss, None, None, None
        else:
            return tf.nn.compute_average_loss(loss, global_batch_size=bs), \
                   tf.nn.compute_average_loss(permu_loss, global_batch_size=batch_size*action_refinement), \
                   tf.nn.compute_average_loss(align_loss, global_batch_size=batch_size*action_refinement), \
                   sampled_permus, tf.stop_gradient(log_potentials), advantages

    #@tf.function
    def custom_gather(a, b):
        """ Permutes the vectors in the last dimension of a according to
            the corresponding vectors in the last dimension of b
            e.g. a=[[1,4,7,6],[4,3,6,8]], b=[[0,3,1,2],[2,1,3,0]]
            Returns [[1,6,4,7],[6,3,8,4]]
        """
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
    
    def pg_loss_function(pretrain_done, kl_coeff, sampled_permus, 
                         log_potentials, advantages, b):
        # computes permutation loss by only forward passing
        # through permutation transformer
        permu_inputs = prepare_batch_for_pt(pretrain_done, 
                           action_refinement, b)        
        _, activations, kl, log_nom, log_denom = \
            order(permu_inputs, training=True)    
        if pg_final_layer == 'sinkhorn':
            log_nominator = tf.linalg.trace(tf.matmul(activations, 
                                                      sampled_permus,
                                                      transpose_a=True))
            log_nominator = tf.reshape(log_nominator, (-1, 1))  
            ratio = tf.math.exp(log_nominator - log_denom - log_potentials)
        elif pg_final_layer == 'plackett':            
            onedim_sampled_permus = tf.cast(tf.math.argmax(sampled_permus, axis=-1), tf.int32)
            exp_actis = custom_gather(activations, onedim_sampled_permus)
            exp_actis = tf.math.exp(exp_actis)
            reverse_cumsum_exp_actis = tf.math.cumsum(exp_actis[:, ::-1], axis=-1)[:, ::-1]
            eps = 1e-20
            log_nominator = tf.math.log(exp_actis + eps) - tf.math.log(reverse_cumsum_exp_actis + eps)
            if dataset == 'captioning':
                token_ind = b['token_indicators'][:, 1:]
            elif dataset in ['wmt', 'django', 'gigaword']:
                token_ind = b['decoder_token_indicators'][:, 1:]
            repeated_tokeninds = tf.repeat(token_ind, action_refinement, axis=0)
            log_nominator = tf.reduce_sum(log_nominator[:, 1:] * repeated_tokeninds, axis=-1, keepdims=True)   
            #tf.print(exp_actis[0], reverse_cumsum_exp_actis[0], (log_nominator * repeated_tokeninds)[0], summarize=-1)
            ratio = tf.math.exp(log_nominator - log_potentials)
        reward = tf.math.minimum(ratio * advantages, 
                                 tf.clip_by_value(ratio, 0.9, 1.1) * advantages)
        #tf.print("action prob ratio", tf.squeeze(ratio), summarize=-1)
        permu_loss = tf.squeeze(-reward + kl_coeff * kl)            
        return tf.nn.compute_average_loss(permu_loss, global_batch_size=batch_size*action_refinement)
    
    def dummy_loss_function(b):
        # process the dataset batch dictionary into the standard
        # model input format
        inputs, permu_inputs = prepare_permutation(
            b, vocabs[-1].size(),
            'l2r' if order == 'sao' else order, dataset, policy_gradient,
            tf.constant(True), tf.constant(action_refinement), decoder=model)
        # The "call" function of the decoder is a dummy function. 
        # During training, we invoke model.loss, which sums the 
        # returned loss across all submodules;
        # During testing, we invoke model.beam_search, which 
        # invokes the beam search function on all submodules;
        # Thus the "call" function of the decoder is only used to 
        # build the parameters and is only called once
        # (it invokes all the "call" functions of submodules, which 
        # is required for tf.keras to know the parameters of the model).
        _ = model(inputs)
        loss, inputs = model.loss(inputs, training=True)

    @tf.function(input_signature=[train_dataset.element_spec])
    def wrapped_dummy_loss_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        strategy.run(dummy_loss_function, args=(b,))
    
    # build the weights of the model using an initial forward pass    
    for batch in train_dataset:
        if dataset in ['wmt', 'django', 'gigaword']:
            if strategy.num_replicas_in_sync > 1:
                be = tf.concat(batch["encoder_words"].values, axis=0)
                bd = tf.concat(batch["decoder_words"].values, axis=0)
            else:
                be = batch["encoder_words"]
                bd = batch["decoder_words"]
            enc = tf.strings.reduce_join(vocabs[0].ids_to_words(be), axis=1, separator=' ')
            dec = tf.strings.reduce_join(vocabs[-1].ids_to_words(bd), axis=1, separator=' ')
            for z in range(len(enc)):
                print(enc[z])
                print(dec[z])
        wrapped_dummy_loss_function(batch)
        break

    print("----------Done initializing (but not loading) the weights of the model-----------")

    def decode_function(b):
        # calculate the ground truth sequence for this batch; and
        # perform beam search using the current model
        # show several model predicted sequences and their likelihoods
        if dataset in ['captioning', 'gigaword']:
            maxit = 30
        elif dataset in ['wmt', 'django']:
            maxit = 100
        inputs = prepare_batch_for_lm(tf.constant(1), b)
#         if isinstance(order, tf.keras.Model):
#             permu_inputs = prepare_batch_for_pt(tf.constant(True), tf.constant(1), b)
#             permu_outputs, _, _, _, _ = order(permu_inputs)        
        out = tf.strings.reduce_join(
            vocabs[-1].ids_to_words(inputs[4]), axis=1, separator=' ')
        cap, logp = beam_search(
            inputs, model, dataset, beam_size=beam_size, max_iterations=maxit)
        cap = tf.strings.reduce_join(
            vocabs[-1].ids_to_words(cap), axis=2, separator=' ')
        
        for i in range(tf.shape(cap)[0]):
            tf.print("Label:", out[i])
            for j in range(beam_size):
                tf.print("[p =", tf.math.exp(logp[i, j]), 
                         "] Model:", cap[i, j])
        
#         # calculate hamming distance between common first permutation and PT permutation
#         decoder_pos = get_permutation(b['token_indicators'], b['words'], "common")
        
#         def get_1dim_permu(x):
#             # given 2-dim int32 permutation, get its 1-dim representation
#             locs = tf.where(x == 1)
#             d1 = tf.shape(locs)[0]
#             d2 = tf.shape(locs)[1]
#             locs = tf.reshape(locs, [locs[-1,0]+1, d1//tf.cast(locs[-1,0]+1, tf.int32), d2])
#             return locs[:, :, -1][:, tf.newaxis, :]
        
#         dec_1dim = get_1dim_permu(tf.cast(decoder_pos, tf.int32))
#         pt_1dim = get_1dim_permu(tf.cast(permu_outputs, tf.int32))
        
#         tf.print("Decoder & PT permutations", 
#                  tf.concat([dec_1dim, pt_1dim], axis=1), 
#                  summarize=-1)
        
#         def to_sparse(x):
#             idx = tf.where(tf.not_equal(x, 0))
#             return tf.SparseTensor(idx, tf.gather_nd(x, idx), tf.cast(tf.shape(x), tf.int64))
        
#         dec_1dim_sqz = tf.squeeze(dec_1dim)
#         pt_1dim_sqz = tf.squeeze(pt_1dim)
#         e_dist = tf.edit_distance(to_sparse(dec_1dim_sqz), 
#                                   to_sparse(pt_1dim_sqz), 
#                                   normalize=False)
#         e_dist = e_dist / tf.reduce_sum(b['token_indicators'][:, 1:], axis=1)
#         tf.print("Edit distance from decodeer PT", e_dist, summarize=-1)
#         tf.print("Avg edit distance", tf.reduce_mean(e_dist))

    @tf.function(input_signature=[train_dataset.element_spec])
    def wrapped_decode_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        strategy.run(decode_function, args=(b,))

    # restore an existing model if one exists and create a directory
    # if the ckpt directory does not exist
    modified_ckpt = model_ckpt[:-3]+"_ckpt.h5"
    tf.io.gfile.makedirs(os.path.dirname(modified_ckpt))
    
    if tf.io.gfile.exists(modified_ckpt.replace(".", ".pt.")) and isinstance(order, tf.keras.Model):
        order.load_weights(modified_ckpt.replace(".", ".pt."))       
    if tf.io.gfile.exists(modified_ckpt):
        model.load_weights(modified_ckpt)
        
    best_loss = 999999.0
    vars = model.trainable_variables
    pt_vars = order.trainable_variables \
        if isinstance(order, tf.keras.Model) else []


        
    def step_function_no_pg(pretrain_done, kl_coeff, b):
        with tf.GradientTape(persistent=True) as tape:
            loss, _, _, _, _, _ = loss_function(pretrain_done, kl_coeff, b)            
        grads = tape.gradient(loss, vars + pt_vars)
        optim.apply_gradients(list(zip(grads[:len(vars)], vars)))
        pt_optim.apply_gradients(list(zip(grads[len(vars):], pt_vars)))
        #tf.print("optim.lr", optim.lr)
        del tape
        return loss
        
    def step_function_normal_pg(pretrain_done, kl_coeff, b):
        # performing a gradient descent step on a batch of data
        with tf.GradientTape(persistent=True) as tape:
            loss, permu_loss, align_loss, sampled_permus, log_potentials, advantages \
                = loss_function(pretrain_done, kl_coeff, b) 
        grads = tape.gradient(loss, vars)
        optim.apply_gradients(list(zip(grads, vars)))
        #tf.print("optim.lr", optim.lr, "pt_optim.lr", pt_optim.lr)
        grads = tape.gradient(permu_loss, pt_vars)
        pt_optim.apply_gradients(list(zip(grads, pt_vars)))
        grads = tape.gradient(align_loss, pt_vars[:num_emb_matrices])
        pt_optim.apply_gradients(list(zip(grads, pt_vars[:num_emb_matrices])))  
        del tape
        return loss
    
    def step_function_ppo_pg(pretrain_done, kl_coeff, b):
        # performing a gradient descent step on a batch of data
        with tf.GradientTape(persistent=True) as tape:
            loss, permu_loss, align_loss, sampled_permus, log_potentials, advantages \
                = loss_function(pretrain_done, kl_coeff, b) 
        grads = tape.gradient(loss, vars)
        optim.apply_gradients(list(zip(grads, vars)))
        grads = tape.gradient(align_loss, pt_vars[:num_emb_matrices])
        pt_optim.apply_gradients(list(zip(grads, pt_vars[:num_emb_matrices])))
        del tape
        #tf.print("optim.lr", optim.lr, "pt_optim.lr", pt_optim.lr)
        for _ in range(3):
            with tf.GradientTape() as tape:
                permu_loss =  pg_loss_function(pretrain_done, kl_coeff, sampled_permus, 
                                               log_potentials, advantages, b)   
            grads = tape.gradient(permu_loss, pt_vars)
            #tf.print("--------------------emb before update", order.values_embedding(tf.constant([5]))[0, :10])
            pt_optim.apply_gradients(list(zip(grads, pt_vars)))
            #tf.print("--------------------emb after update", order.values_embedding(tf.constant([5]))[0, :10])
        return loss    
    
    def step_function_decoderonly(pretrain_done, kl_coeff, b):
        # performing a gradient descent step on a batch of data
        with tf.GradientTape() as tape:
            loss, permu_loss, align_loss, sampled_permus, log_potentials, advantages \
                = loss_function(pretrain_done, kl_coeff, b)
        grads = tape.gradient(loss, vars)
        optim.apply_gradients(list(zip(grads, vars)))      
        #tf.print("training decoder only: optim.lr", optim.lr)
        return loss    
    
    def step_function_ptonly_normal(pretrain_done, kl_coeff, b):
        # performing a gradient descent step on a batch of data
        with tf.GradientTape(persistent=True) as tape:
            loss, permu_loss, align_loss, sampled_permus, log_potentials, advantages \
                = loss_function(pretrain_done, kl_coeff, b)
        grads = tape.gradient(permu_loss, pt_vars)
        pt_optim.apply_gradients(list(zip(grads, pt_vars)))  
        grads = tape.gradient(align_loss, pt_vars[:num_emb_matrices])
        pt_optim.apply_gradients(list(zip(grads, pt_vars[:num_emb_matrices])))
        #tf.print("training permutation only: pt_optim.lr", pt_optim.lr)
        return loss        
    
    def step_function_ptonly_ppo(pretrain_done, kl_coeff, b):
        # performing a gradient descent step on a batch of data
        with tf.GradientTape() as tape:
            loss, permu_loss, align_loss, sampled_permus, log_potentials, advantages \
                = loss_function(pretrain_done, kl_coeff, b)
        grads = tape.gradient(align_loss, pt_vars[:num_emb_matrices])
        pt_optim.apply_gradients(list(zip(grads, pt_vars[:num_emb_matrices])))
        del tape
        for _ in range(3):
            with tf.GradientTape() as tape:
                permu_loss =  pg_loss_function(pretrain_done, kl_coeff, sampled_permus, 
                                               log_potentials, advantages, b)           
            grads = tape.gradient(permu_loss, pt_vars)
            pt_optim.apply_gradients(list(zip(grads, pt_vars)))                
                
        #tf.print("training permutation only: pt_optim.lr", pt_optim.lr)
        return loss            

    if policy_gradient != 'without_bvn':
        step_function = step_function_no_pg
    else:
        if use_ppo:
            step_function = step_function_ppo_pg
        else:
            step_function = step_function_normal_pg
            
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  train_dataset.element_spec])     
    def wrapped_step_function(pretrain_done, kl_coeff, b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        result = strategy.run(step_function, args=(pretrain_done, kl_coeff, b))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, result, axis=None)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  train_dataset.element_spec])     
    def wrapped_step_decoderonly(pretrain_done, kl_coeff, b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        result = strategy.run(step_function_decoderonly, 
                              args=(pretrain_done, kl_coeff, b))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, result, axis=None)    

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  train_dataset.element_spec])     
    def wrapped_step_ptonly_normal(pretrain_done, kl_coeff, b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        result = strategy.run(step_function_ptonly_normal, 
                              args=(pretrain_done, kl_coeff, b))        
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, result, axis=None)        

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.bool),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  train_dataset.element_spec])     
    def wrapped_step_ptonly_ppo(pretrain_done, kl_coeff, b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        result = strategy.run(step_function_ptonly_ppo, 
                              args=(pretrain_done, kl_coeff, b))            
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, result, axis=None)       
    
    if use_ppo:
        wrapped_step_ptonly = wrapped_step_ptonly_ppo
    else:
        wrapped_step_ptonly = wrapped_step_ptonly_normal

    # training for a pre specified number of epochs while annealing
    # the learning rate linearly towards zero
    iteration = -1
    step_modes = [wrapped_step_decoderonly, wrapped_step_ptonly]
    step_freqs = alternate_training
    step_mode = 0
    cur_step_count = 0
    
    if dataset == 'captioning':
        maxlength = 45
    elif dataset == 'gigaword':
        maxlength = 30
    elif dataset in ['wmt']:
        maxlength = 100
    elif dataset in ['django']:
        maxlength = 40
        
    def save_weights(model_ckpt):
        model.save_weights(model_ckpt)
        if isinstance(order, tf.keras.Model):
            order.save_weights(model_ckpt.replace(".", ".pt."))

        symbolic_weights = getattr(optim, 'weights')
        weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
        with open(model_ckpt[:-3] + "_optim.obj", "wb") as f:
            pickle.dump(weight_values, f)
        if isinstance(order, tf.keras.Model):
            symbolic_weights = getattr(pt_optim, 'weights')
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(model_ckpt[:-3] + "_pt_optim.obj", "wb") as f:
                pickle.dump(weight_values, f)        
                
    # dataset size
    if dataset == 'gigaword':
        d_size = 3803897
    elif dataset == 'wmt':
        d_size = 609324
    elif dataset == 'captioning':
        d_size = 591435
    elif dataset == 'django':
        d_size = 16000

    # create data frames for global sequence-level statistics
    training_time_df = pd.DataFrame(columns=[
        'Model',
        'Type',
        'Time'])
        
        
    for epoch in range(num_epoch):
        dist_it = iter(train_dataset)
        def get_cur_lr(lr):
            if iteration + 1 < warmup:
                return lr * (iteration + 1) / max(warmup, 1)
            else:
                if lr_schedule == 'constant':
                    return lr
                elif lr_schedule == 'linear':
                    g_its = num_epoch * d_size / batch_size - warmup
                    return lr * (g_its - min(iteration - warmup + 1, g_its)) / g_its
        
        # loop through the entire dataset once (one epoch)
        for batch in train_dataset:
            if kl_log_linear > 0:
                log_kl_start = np.log(kl_coeff)
                log_kl_end = np.log(kl_log_linear)                
                if dataset in ['gigaword', 'wmt', 'django', 'captioning']:
                    g_its = num_epoch * d_size / batch_size
                    kl_coeff = log_kl_start + (log_kl_end - log_kl_start) * min(iteration + 1, g_its) / g_its            
                else:
                    kl_coeff = log_kl_start + (log_kl_end - log_kl_start) * epoch / (num_epoch - 1)
                kl_coeff = np.exp(kl_coeff).astype(np.float32)
                print("Current kl", kl_coeff)
            optim.lr.assign(get_cur_lr(init_lr))
            pt_optim.lr.assign(get_cur_lr(pt_init_lr))
            if dataset == 'captioning':
                words = batch["words"]
            elif dataset in ['wmt', 'django', 'gigaword']:
                words = batch["decoder_words"]
            if (strategy.num_replicas_in_sync == 1 and tf.shape(words)[1] > maxlength) \
                or (strategy.num_replicas_in_sync > 1 and tf.shape(words.values[0])[1] > maxlength):
                continue
            if (strategy.num_replicas_in_sync == 1 and tf.shape(words)[0] < 3) \
                or (strategy.num_replicas_in_sync > 1 and tf.shape(tf.concat(words.values, axis=0))[0] < 3*strategy.num_replicas_in_sync) \
                or (strategy.num_replicas_in_sync > 5 and tf.shape(tf.concat(words.values, axis=0))[0] % strategy.num_replicas_in_sync != 0):
                continue                
            iteration += 1
            cur_step_count += 1

            # to load optimizer state, run a dummy iteration
            # so that the optimizers are tracking the correct
            # number of parameters  
            if iteration == 0 and tf.io.gfile.exists(modified_ckpt):    
                optim.lr.assign(0.0)
                pt_optim.lr.assign(0.0)

            if iteration <= decoder_pretrain:
                pt_optim.lr.assign(0.0)
            elif iteration > 0:
                pt_optim.lr.assign(get_cur_lr(pt_init_lr))

            start_time = time.time()
            if not alternate_training:
                wsf = wrapped_step_function
            else:
                wsf = step_modes[step_mode]
            print("It: {} Train Loss: {}".format(
                iteration, wsf(tf.constant(
                    iteration > decoder_pretrain), tf.constant(kl_coeff), batch)))
            end_time = time.time()

            if iteration > 0:
                training_time_df = training_time_df.append({
                    "Model": model_ckpt,
                    "Type": order.upper() if isinstance(order, str) else "VOI",
                    'Word': end_time - start_time},
                    ignore_index=True)
                training_time_df.to_csv(f'{model_ckpt}_training_time_df.csv')

            # if alternate training, check whether to change step mode
            if alternate_training and cur_step_count >= step_freqs[step_mode]:
                step_mode = 1 - step_mode
                cur_step_count = 0

            # save every 2k training steps (not a separate model snapshot)
            if iteration > 0 and iteration % 2000 == 0:
                save_weights(model_ckpt[:-3] + "_ckpt.h5") 

            if iteration % 100 == 0:
                wrapped_decode_function(batch)

            if iteration == 0 and tf.io.gfile.exists(modified_ckpt):   
                if os.path.exists(model_ckpt[:-3] + "_ckpt_optim.obj"):
                    with open(model_ckpt[:-3] + "_ckpt_optim.obj", "rb") as f:
                        weight_values = pickle.load(f)
                    optim.set_weights(weight_values)        
                if os.path.exists(model_ckpt[:-3] + "_ckpt_pt_optim.obj"):
                    with open(model_ckpt[:-3] + "_ckpt_pt_optim.obj", "rb") as f:
                        weight_values = pickle.load(f)
                    if alternate_training:
                        _ = step_modes[1](tf.constant(
                                iteration > decoder_pretrain), tf.constant(kl_coeff), batch)
                    pt_optim.set_weights(weight_values)                   
                optim.lr.assign(get_cur_lr(init_lr))
            if iteration == 0:
                save_weights(model_ckpt[:-3] + "_init.h5")
            if save_interval > 0 and iteration % save_interval == 0 and iteration > 0:
                save_weights(model_ckpt[:-3] + "_iteration{}.h5".format(iteration))
        
        # save the model at the current epoch
        save_weights(model_ckpt[:-3] + "_ckpt.h5")
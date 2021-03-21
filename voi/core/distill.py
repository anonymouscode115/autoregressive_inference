from voi.data.load_captioning import faster_rcnn_dataset
from voi.data.load_wmt import wmt_dataset
from voi.nn.input import TransformerInput
from voi.nn.input import RegionFeatureInput
from voi.algorithms.beam_search import beam_search
from voi.birkoff_utils import birkhoff_von_neumann
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import get_permutation
import nltk
import tensorflow as tf
import os
import numpy as np

wmt_batch_spec = [{
    'encoder_words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'encoder_token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    'decoder_words': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    'decoder_token_indicators': tf.TensorSpec(shape=[None, None], dtype=tf.float32)}]

@tf.function(input_signature=wmt_batch_spec)
def prepare_batch_wmt(batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:
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
    batch_size = tf.shape(mask)[0]
    
    return [words[:, :-1], encoder_words,
            tf.greater(mask[:, :-1], 0), tf.greater(encoder_token_ind, 0),
            words[:, 1:], None, None, None, None, None, None, tf.zeros([batch_size]),
            tf.zeros([batch_size]), tf.zeros([batch_size]), 
            tf.zeros([batch_size]), tf.zeros([batch_size, 1])]

def prepare_permutation(batch,
                        dataset,
                        tgt_vocab_size):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    dataset: str
        type of dataset (captioning or wmt)        
    tgt_vocab_size: tf.Tensor
        the number of words in the target vocabulary of the model; used in order
        to calculate labels for the language model logits

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # process the dataset batch dictionary into the standard
    # model input format
    if dataset in ['wmt', 'django', 'gigaword']:
        prepare_batch = prepare_batch_wmt
    inputs = prepare_batch(batch)

    # the order is fixed
    if dataset in ['wmt', 'django', 'gigaword']:
        bt, bw = batch['decoder_token_indicators'], batch['decoder_words']
    inputs[5] = get_permutation(bt, bw, tf.constant('l2r'))

    # convert the permutation to absolute and relative positions    
    inputs[6] = inputs[5][:, :-1, :-1]
    inputs[7] = permutation_to_relative(inputs[5])

    # convert the permutation to label distributions
    # also records the partial absolute position at each decoding time step
    hard_pointer_labels, inputs[10] = permutation_to_pointer(inputs[5][:, tf.newaxis, :, :])
    inputs[8] = tf.squeeze(hard_pointer_labels, axis=1)
    inputs[9] = tf.matmul(inputs[5][
        :, 1:, 1:], tf.one_hot(inputs[4], tf.cast(tgt_vocab_size, tf.int32)))

    return inputs    
    
def distill_dataset(tfrecord_folder,
                     batch_size,
                     beam_size,
                     model,
                     model_ckpt,
                     vocabs,
                     dataset_type,
                     strategy,
                     save_path):
    """Sequence-level dataset distillation using beam search
    
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
        the caption model to be distilled; an instance of Transformer that
        returns a data class TransformerInput
    model_ckpt: str
        the path to an existing model checkpoint
    vocabs: list of Vocabulary
        the model vocabulary which contains mappings
        from words to integers
    dataset: str
        type of dataset (captioning or wmt)        
    strategy: tf.distribute.Strategy
        the strategy to use when distributing a model across many gpus
        typically a Mirrored Strategy        
    save_path: str
        save path for distillation output """

    # create a distillation pipeline
    dataset = wmt_dataset(tfrecord_folder, batch_size, shuffle=False)
    prepare_batch = prepare_batch_wmt
        
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    def dummy_loss_function(b):
        # process the dataset batch dictionary into the standard
        # model input format
        inputs = prepare_permutation(b, dataset_type, vocabs[-1].size())
        _ = model(inputs)
        loss, _ = model.loss(inputs, training=True)
        loss = tf.zeros([batch_size])
        return tf.nn.compute_average_loss(loss, global_batch_size=batch_size)

    @tf.function(input_signature=[dataset.element_spec])
    def wrapped_dummy_loss_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function using data parallelism
        result = strategy.run(dummy_loss_function, args=(b,))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, result, axis=None)  
    
    def decode_function(b):
        # perform beam search using the current model and also
        # get the log probability of sequence
        if dataset_type in ['wmt', 'django']:
            maxit = 150        
        elif dataset_type in ['gigaword']:
            maxit = 40
        inputs = prepare_batch(b)
        cap, logp = beam_search(
            inputs, model, dataset_type, beam_size=beam_size, max_iterations=maxit)
        cap = tf.strings.reduce_join(
            vocabs[-1].ids_to_words(cap), axis=2, separator=' ')
        src = tf.strings.reduce_join(
            vocabs[-1].ids_to_words(inputs[1]), axis=1, separator=' ')
        return src, cap, logp

    @tf.function(input_signature=[dataset.element_spec])               
    def wrapped_decode_function(b):
        # distribute the model across many gpus using a strategy
        # do this by wrapping the loss function
        return strategy.run(decode_function, args=(b,))

    # run the model for a single forward pass
    # and load en existing checkpoint into the trained model
    for batch in dataset:
        wrapped_dummy_loss_function(batch)
        break
        
    print("----------Done initializing the weights of the model-----------") 
    model.load_weights(model_ckpt)
    print("----------Done loading the weights of the model-----------")    
        
    # loop through the entire dataset once (one epoch)
    b_idx = 0
    
    f1 = open(os.path.join(save_path, "src_distillation.BPE.txt"), "w")
    f2 = open(os.path.join(save_path, "tgt_distillation.BPE.txt"), "w")
    
    # eliminate all elements in the array whose 
    # batch dimension is zero
    def eliminate_empty(arr):
        result = []
        for x in arr:
            if x.shape[0] != 0:
                result.append(x)
        return result
        
    def parse_output(s):
        return s.decode("utf-8").replace(
                    "<pad>", "").replace("<start>", "").replace(
                    "<end>", "").replace("  ", " ").strip()
        
    for batch in dataset:
        print("Batch index", b_idx)
        b_idx += 1

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        src, cap, log_p = wrapped_decode_function(batch)
        if strategy.num_replicas_in_sync == 1:
            src = src.numpy()
            cap = cap.numpy()
        else:
            # when evaluating on multi gpus, the data might be distributed
            # in a way such that some gpus receive empty inputs, 
            # i.e. the batch dimension is zero
            src = tf.concat(eliminate_empty(src.values), axis=0).numpy()
            cap = tf.concat(eliminate_empty(cap.values), axis=0).numpy()
            log_p = tf.concat(eliminate_empty(log_p.values), axis=0)

        # format the model predictions into a string
        for i in range(cap.shape[0]):
            if dataset_type in ['wmt', 'django', 'gigaword']:
                model_sentence = parse_output(cap[i, 0])
                print("{}: [p = {}] {}".format(i, 
                                               np.exp(log_p[i, 0].numpy()),
                                               model_sentence))
                print(parse_output(src[i]), file=f1)
                print(model_sentence, file=f2)
                
        f1.flush()
        f2.flush()
   
    f1.close()
    f2.close()
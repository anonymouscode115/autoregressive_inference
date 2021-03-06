from voi.data.load import faster_rcnn_dataset
from voi.nn.input import TransformerInput
from voi.nn.input import RegionFeatureInput
from voi.algorithms.beam_search import beam_search
from voi.birkoff_utils import birkhoff_von_neumann
from voi.permutation_utils import permutation_to_pointer
from voi.permutation_utils import permutation_to_relative
from voi.permutation_utils import get_permutation
import tensorflow as tf
import os


def prepare_batch_for_pt(batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    vocab_size: tf.Tensor
        the number of words in the vocabulary of the model; used in order
        to calculate labels for the language model logits

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

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        queries=words,
        values=region,
        queries_mask=tf.logical_not(start_end_or_pad),
        values_mask=tf.greater(image_ind, 0))

    return inputs


def pretrain_faster_rcnn_dataset(train_folder,
                                 validate_folder,
                                 batch_size,
                                 num_epoch,
                                 model,
                                 model_ckpt,
                                 order,
                                 vocab):
    """Trains a transformer based caption model using features extracted
    using a facter rcnn object detection model

    Arguments:

    train_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk;
        used for training
    validate_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk;
        used for validation
    batch_size: int
        the maximum number of training examples in a
        single batch
    num_epochs: int
        the number of loops through the entire dataset to
        make before termination
    model: Decoder
        the caption model to be trained; an instance of Transformer that
        returns a data class TransformerInput
    model_ckpt: str
        the path to an existing model checkpoint or the path
        to be written to when training
    order: str
        the autoregressive ordering to train Transformer-InDIGO using;
        l2r or r2l for now, will support soft orders later
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers"""

    pt_ckpt = os.path.join(os.path.dirname(
        model_ckpt), "pt_" + os.path.basename(model_ckpt))

    # create a training pipeline
    init_lr = 0.00005
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
    train_dataset = faster_rcnn_dataset(train_folder, batch_size)
    validate_dataset = faster_rcnn_dataset(validate_folder,
                                           batch_size, shuffle=False)

    def loss_function(it, b, verbose=False):

        # process the dataset batch dictionary into the standard
        # model input format
        mat = model(prepare_batch_for_pt(b), training=True)
        labels = get_permutation(
            b['token_indicators'], b['words'], tf.constant(order))
        loss = tf.reduce_mean((labels - mat) ** 2)
        if verbose:
            print('It: {} Train Loss: {}'.format(it, loss))
        return loss

    def decode(b):

        # calculate the ground truth sequence for this batch; and
        # sample a permutation using the model
        # show several permutations and the sentences
        inputs = prepare_batch_for_pt(b)
        out = tf.strings.reduce_join(
            vocab.ids_to_words(b['words']), axis=1, separator=' ')
        mat = model(inputs, training=True)
        for i in range(out.shape[0]):
            print("Label: {}".format(out[i].numpy().decode('utf8')))
            print("Permutation:\n{}".format(mat[i].numpy()))

    def validate():

        # accumulate the validation loss across the entire dataset
        # weight the loss by the batch size and normalize
        # the loss to an expected value
        denom, loss = 0.0, 0.0
        for b in validate_dataset:
            n = tf.cast(tf.shape(b['words'])[0], tf.float32)
            denom, loss = denom + n, loss + n * loss_function(0, b)
        return loss / denom

    # run an initial forward pass using the model in order to build the
    # weights and define the shapes at every layer
    for batch in train_dataset.take(1):
        loss_function(-1, batch, verbose=False)

    # restore an existing model if one exists and create a directory
    # if the ckpt directory does not exist
    tf.io.gfile.makedirs(os.path.dirname(model_ckpt))
    if tf.io.gfile.exists(pt_ckpt + ".meta"):
        model.load_weights(pt_ckpt)

    # set up variables for early stopping; only save checkpoints when
    # best validation loss has improved
    best_loss = validate()
    var_list = model.trainable_variables

    # training for a pre specified number of epochs while also annealing
    # the learning rate linearly towards zero
    iteration = 0
    for epoch in range(num_epoch):

        # loop through the entire dataset once (one epoch)
        for batch in train_dataset:

            # keras requires the loss be a function
            optim.minimize(lambda: loss_function(
                iteration, batch, verbose=True), var_list)
            if iteration % 100 == 0:
                decode(batch)

            # increment the number of training steps so far; note this
            # does not save with the model and is reset when loading a
            # pre trained model from the disk
            iteration += 1

        # anneal the model learning rate after an epoch
        optim.lr.assign(init_lr * (1 - (epoch + 1) / num_epoch))

        # normalize the validation loss per validation example
        validation_loss = validate()
        print('It: {} Val Loss: {}'.format(iteration, validation_loss))

        # save once at the end of every epoch; but only save when
        # the validation loss becomes smaller
        if best_loss > validation_loss:
            best_loss = validation_loss
            model.save_weights(pt_ckpt, save_format='tf')

import tensorflow as tf
import os
import pickle as pkl

def wmt_dataset(tfrecord_folder,
                        batch_size,
                        shuffle=True):
    """Builds an input data pipeline for training deep image
    captioning models using region features

    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    batch_size: int
        the maximum number of training examples in a
        single batch
    shuffle: bool
        specifies whether to shuffle the training dataset or not;
        do not shuffle during validation

    Returns:

    dataset: tf.data.Dataset
        a dataset that can be iterated over"""

    # select all files from the disk that contain training examples
    record_files = tf.data.Dataset.list_files(
        os.path.join(tfrecord_folder, "*.tfrecord"))

    # in parallel read from the disk into training examples
    dataset = record_files.interleave(
        parse_wmt_tf_records,
        cycle_length=tf.data.experimental.AUTOTUNE,
        block_length=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,)

    # shuffle and pad the data into batches for training
    if shuffle:
        dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "encoder_words": [None],
        "encoder_token_indicators": [None],
        "decoder_words": [None],
        "decoder_token_indicators": [None]})

    # this line makes data processing happen in parallel to training
    return dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)


def parse_wmt_tf_records(record_files):
    """Parse a list of tf record files into a dataset of tensors for
    training a caption model

    Arguments:

    record_files: list
        a list of tf record files on the disk

    Returns:

    dataset: tf.data.Dataset
        create a dataset for parallel training"""

    return tf.data.TFRecordDataset(record_files).map(
        parse_wmt_sequence_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

def parse_wmt_sequence_example(sequence_example):
    """Parse a single sequence example that was serialized
    from the disk and build a tensor

    Arguments:

    sequence_example: tf.train.SequenceExample
        a single training example on the disk

    Returns:

    out: dict
        a dictionary containing all the features in the
        sequence example"""

    # read the sequence example binary
    context, sequence = tf.io.parse_single_sequence_example(
        sequence_example,
        sequence_features={
            "src_words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "tgt_words": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)})

    # create a padding mask
    src_token_indicators = tf.ones(tf.shape(sequence["src_words"]), dtype=tf.float32)
    tgt_token_indicators = tf.ones(tf.shape(sequence["tgt_words"]), dtype=tf.float32)

    # cast every tensor to the appropriate data type
    src_words = tf.cast(sequence["src_words"], tf.int32)
    tgt_words = tf.cast(sequence["tgt_words"], tf.int32)

    # build a dictionary containing all features
    return dict(
        encoder_words=src_words,
        encoder_token_indicators=src_token_indicators,
        decoder_words=tgt_words,
        decoder_token_indicators=tgt_token_indicators)
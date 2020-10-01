import tensorflow as tf
import os

def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.compat.v1.get_default_session()
    if sess is None:
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.InteractiveSession(config=config)
    return sess
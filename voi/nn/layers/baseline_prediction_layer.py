from voi.nn.wrappers.layer import Layer
from voi.nn.base.block import Block
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class BaselinePredictionLayer(Layer):

    def __init__(self,
                 hidden_size,
                 **kwargs):
        """
        Arguments:
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer """
        
        super(BaselinePredictionLayer, self).__init__()

        self.dense = tf.keras.layers.Dense(1,
                                    activation=None,
                                    **kwargs)
        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        [queries, values, queries_mask, values_mask,
         pretrain_done, _, _, _, _, _, _, _, _,
         object_detections, object_features, object_boxes] = inputs   
        
        baseline_values = self.dense(queries)
        baseline_values = baseline_values * tf.cast(queries_mask[..., tf.newaxis],
                                                    tf.float32)
        baseline_values = tf.squeeze(baseline_values)
        baseline_values = tf.reduce_sum(baseline_values, axis=1)
 
        return baseline_values

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process
        Returns:
        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      ** self.kwargs)

        base_config = super(BaselinePredictionLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
from voi.nn.wrappers.sequential import Sequential
from voi.nn.layers.encoder_layer import EncoderLayer
from voi.nn.layers.encoder_with_position_layer import EncoderWithPositionLayer
from voi.nn.layers.encoder_with_positional_attention_layer import EncoderWithPositionalAttentionLayer
from voi.nn.layers.decoder_layer import DecoderLayer
from voi.nn.layers.decoder_with_positional_attention_layer import DecoderWithPositionalAttentionLayer
from voi.nn.layers.decoder_with_position_layer import DecoderWithPositionLayer
from voi.nn.layers.permutation_layer import PermutationLayer
from voi.nn.layers.permutation_sinkhorn import PermutationSinkhornLayer
from voi.nn.layers.permutation_plackett import PermutationPlackettLayer
from voi.nn.features.discrete_feature import DiscreteFeature
from voi.nn.features.continuous_feature import ContinuousFeature
from voi.nn.features.region_feature import RegionFeature


class PermutationTransformer(Sequential):

    def __init__(self,
                 hidden_size,
                 heads,
                 num_layers,
                 policy_gradient,
                 src_embedding,
                 tgt_embedding,  
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 first_layer='region',
                 pg_final_layer='plackett',
                 pt_positional_attention=True,
                 pt_relative_embedding=False,
                 temperature=1.,
                 dataset='captioning',
                 hungarian_op_path='',
                 **kwargs):
        """Creates a Transformer Keras model for processing sequences
        and uses the tf.layers.Sequential as backend

        Arguments:
        
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        num_layers: int
            the number of variables in the encoder and the decoder modules
            each layer consists of attention residual connections
        policy_gradient: str
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
        queries_embedding: tf.keras.layers.Embedding
            the queries embedding shared between the decoder
            and the permutation transformer
            in image captioning, this is the source detection
            in translation, this is the source vocab embedding
        values_embedding: tf.keras.layers.Embedding
            the values embedding shared between the decoder
            and the permutation transformer  
            in image captioning, this is the target caption
            in translation, this is the target vocab embedding                    
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        first_layer: class
            specifies the class to use for the first layer in the transformer
            defaults to WordFeature if not specified
        pg_final_layer: class
            if policy gradient is 'without_bvn',
            specifies the class to use for the final layer in the transformer
            defaults to 'plackett' if not specified     
        pt_positional_attention: bool
            whether to use positional attention
        pt_relative_embedding: bool
            whether to use relative embedding instead of positional embedding
        temperature: float
            a positive number to divide the permutation logits by prior
            to applying sinkhorn normalization
        dataset: str
            type of dataset
        hungarian_op_path: str
            the path to the cpu / gpu op of hungarian algorithm (for 
            obtaining hard permutation matrices from soft permutation
            matrices) """

        # TODO: Sequential does not technically support nested inputs
        layers = []
        super(PermutationTransformer, self).__init__(layers)

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        
        # the first layer in the transformer depends on the data modality
        # for image captioning using RCNN features select 'region'
        print("PT sinusoid embedding", not pt_relative_embedding and not pt_positional_attention)
        if first_layer == 'discrete':
            layers.extend([DiscreteFeature(
                hidden_size, 
                self.src_embedding, self.tgt_embedding, mode='pt', 
                decoder_pos_emb=not pt_relative_embedding and not pt_positional_attention,
                **kwargs)])
        if first_layer == 'continuous':
            layers.extend([ContinuousFeature(
                hidden_size,
                self.src_embedding, self.tgt_embedding, mode='pt', 
                decoder_pos_emb=not pt_relative_embedding and not pt_positional_attention,
                **kwargs)])
        if first_layer == 'region':
            layers.extend([RegionFeature(
                hidden_size,
                self.src_embedding, self.tgt_embedding, mode='pt', 
                decoder_pos_emb=not pt_relative_embedding and not pt_positional_attention,
                **kwargs)])

        # the encoder processes values and the decoder processes queries
        # build the encoder first in the stack
        if dataset == 'captioning' or not (pt_relative_embedding or pt_positional_attention):
            layers.extend([EncoderLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])
        elif pt_relative_embedding:
            layers.extend([EncoderWithPositionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, num_pos=10, **kwargs) for _ in range(num_layers)])
        elif pt_positional_attention:
            layers.extend([EncoderWithPositionalAttentionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])            

        # depending on the type of network possibly condition on position
        # build the decoder second in the stack
        if pt_positional_attention:
            layers.extend([DecoderWithPositionalAttentionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])
        elif pt_relative_embedding:
            layers.extend([DecoderWithPositionLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])            
        else:
            layers.extend([DecoderLayer(
                hidden_size, hidden_size * 4, heads,
                queries_dropout=queries_dropout,
                keys_dropout=keys_dropout,
                values_dropout=values_dropout,
                causal=False, **kwargs) for _ in range(num_layers)])            

        # the final layer in the transformer depends on the model purpose
        # to run Transformer-InDIGO select 'indigo'
        if policy_gradient != 'without_bvn':
            layers.extend([PermutationLayer(
                hidden_size, temperature=temperature, **kwargs)])
        else:
            if pg_final_layer == 'sinkhorn':
                layers.extend([PermutationSinkhornLayer(
                    hidden_size * 4, heads,
                    queries_dropout=queries_dropout,
                    keys_dropout=keys_dropout, 
                    hungarian_op_path=hungarian_op_path, **kwargs)])
            elif pg_final_layer == 'plackett':
                layers.extend([PermutationPlackettLayer(
                    hidden_size * 4, heads,
                    queries_dropout=queries_dropout,
                    keys_dropout=keys_dropout, **kwargs)])

        super(PermutationTransformer, self).__init__(layers)

        self.last_layer = layers[-1]
        
        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.policy_gradient = policy_gradient
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.first_layer = first_layer
        self.pg_final_layer = pg_final_layer
        self.pt_positional_attention = pt_positional_attention
        self.pt_relative_embedding = pt_relative_embedding
        self.temperature = temperature
        self.dataset = dataset
        self.kwargs = kwargs

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      heads=self.heads,
                      num_layers=self.num_layers,
                      src_embedding=self.src_embedding,
                      tgt_embedding=self.tgt_embedding,                      
                      policy_gradient=self.policy_gradient,
                      queries_dropout=self.queries_dropout,
                      keys_dropout=self.keys_dropout,
                      values_dropout=self.values_dropout,
                      first_layer=self.first_layer,
                      pg_final_layer=self.pg_final_layer,
                      pt_positional_attention=self.pt_positional_attention,
                      pt_relative_embedding=self.pt_relative_embedding,
                      temperature=self.temperature,
                      dataset=self.dataset,
                      ** self.kwargs)

        base_config = super(PermutationTransformer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))

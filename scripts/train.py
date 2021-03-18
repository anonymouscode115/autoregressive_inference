from voi.core.sess_util import get_session
from voi.core.train import train_dataset
from voi.nn.transformer import Transformer
from voi.nn.baseline_transformer import BaselineTransformer
from voi.nn.permutation_transformer import PermutationTransformer
from voi.process.captions import Vocabulary
import tensorflow as tf
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--train_folder', type=str, default='tfrecords',
        help="Path to training set tfrecords folder")
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help="""
        Batch size for each training iteration. 
        If training VOI, action_refinement permutations 
        are sampled per data, so the actual batch dimension
        has length batch_size * action_refinement.
        """)
    parser.add_argument(
        '--beam_size', type=int, default=1,
        help="Beam size for visualization during training.")
    parser.add_argument(
        '--vocab_file', type=str, nargs='+',
        help="""
        Vocabulary file. Currently implemented as shared between 
        permutation transformer and autoregressive decoder.""")
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help="Number of training epochs.")
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/nsds.h5',
        help="Model checkpoint saving and loading.")
    parser.add_argument(
        '--save_interval', type=int, default=50000,
        help="""
        The model snapshot saving interval.
        The snapshot is named by the # of iterations. 
        Disable if <=0 
        (model_name_ckpt.h5 etc still saved every 2000 iterations)
        """)
    parser.add_argument(
        '--hungarian_op_path', type=str, default='./hungarian.so',
        help="""
        Path to the tensorflow hungarian custom op.
        If the path is invalid or the loading fails, then 
        tf.py_function with scipy.optimize.linear_sum_assignment
        is used, which is not efficient for multi-gpu training.
        """)    
    parser.add_argument(
        '--embedding_size', type=int, default=256,
        help="Size of token embedding.")
    parser.add_argument(
        '--share_embedding', type=str, default="True",
        help="""
        Whether to share the embedding between the 
        permutation transformer and the autoregressive decoder.
        This parameter is only used for VOI training.
        """)
    parser.add_argument(
        '--heads', type=int, default=4,
        help="Number of attention heads")
    parser.add_argument(
        '--num_layers', type=int, default=2,
        help="Number of attention layers")
    parser.add_argument(
        '--queries_dropout', type=float, default=0.1,
        help="Prob of queries dropout in attention")
    parser.add_argument(
        '--keys_dropout', type=float, default=0.1,
        help="Prob of keys dropout in attention")
    parser.add_argument(
        '--values_dropout', type=float, default=0.1,
        help="Prob of values dropout in attention")
    parser.add_argument(
        '--label_smoothing', type=float, default=0.0,
        help="Coefficient for label smoothing")    
    parser.add_argument(
        '--first_layer', type=str,
        default='region', choices=['region', 'discrete', 'continuous'],
        help="""
        First layer of the decoder and permutation transformer.
        For captioning, choose "region".
        For other tasks, choose "discrete".
        """)
    parser.add_argument(
        '--final_layer', type=str,
        default='indigo', choices=['indigo', 'logits'],
        help="""
        Final layer of the autoregressive decoder.
        "indigo" is the Transformer INDIGO with 
        Logits and PointerAfterLogits layers and with
        both token and position losses. It generates
        the target sequences through insertion.
        "logits" is "indigo" without the PointerAfterLogits layer,
        i.e. there is no position loss in this case, and generation
        is left-to-right.
        For this argument, "indigo" covers all functionalities
        of "logits". In practice, when training with fixed orderings,
        select "indigo" for this argument and "l2r" "r2l" "rare" etc
        for the "orders" argument.
        """)
    parser.add_argument(
        '--order', type=str,
        default='soft', 
        choices=['l2r', 'r2l', 'rare', 'common', 'test', 'soft', 'sao'],
        help="""
        The type of autoregressive ordering to train,
        either fixed ordering or VOI.
        l2r = left-to-right;
        r2l = right-to-left;
        rare = rare-first;
        common = common-first;
        soft = VOI;
        sao = searched adaptive order;
        test = debugging.
        """)
    parser.add_argument(
        '--policy_gradient', type=str,
        default='none', choices=['with_bvn', 'without_bvn', 'none'],
        help="""
        Whether to use policy gradient for training.
        This parameter is used for VOI training only.
        Default: none (no policy gradient, which is necessary for
                       fixed ordering training).
        Choices: 
            1. with_bvn = Use policy gradient with probabilities of 
                hard permutations based on Berkhoff von Neumann decomposition
                of doubly stochastic matrix (IN PRACTICE THIS HAS NUMERICAL ISSUES SO 
                WE HAVE DISABLED IT).
            2. without_bvn = After applying the Hungarian algorithm on soft 
                permutation from the Sinkhorn operation to obtain hard permutations, 
                i.e. P = Hungarian(Sinkhorn((X + eps) / self.temperature)),
                where X = q(y, x),
                the probabilities of hard 
                permutations are proportionally based on Gumbel-Matching distribution 
                i.e. exp(<X,P>_F), see https://arxiv.org/abs/1802.08665) 
        """)   
    parser.add_argument(
        '--decoder_pretrain', type=int, default=-1,
        help="""
        Number of batches for decoder pretraining using
        uniformly sampled permutations. Disabled if <0.
        After pretraining, orderings are sampled from
        the permutation transformer.
        """)    
    parser.add_argument(
        '--decoder_init_lr', type=float, default=0.0001,
        help="Decoder initial learning rate")      
    parser.add_argument(
        '--pt_init_lr', type=float, default=0.00001,
        help="Permutation Transformer initial learning rate") 
    parser.add_argument(
        '--lr_schedule', type=str, 
        default='linear', choices=['linear', 'constant'],
        help="Learning rate schedule, either linear decay or constant")
    parser.add_argument(
        '--warmup', type=int, default=0,
        help="Number of warmup iterations")
    parser.add_argument(
        '--pt_pg_type', type=str,
        default='sinkhorn', choices=['plackett', 'sinkhorn'],
        help="""
        Modeling q(.|x, y) as Gumbel-Sinkhorn distribution
        ("sinkhorn") or Plackett-Luce Distribution ("plackett")
        """)
    parser.add_argument(
        '--pt_positional_attention', type=str, default="False",
        help="""
        Whether to use Transformer-XL relative position encoding
        for the permutation transformer.
        """)
    parser.add_argument(
        '--pt_relative_embedding', type=str, default="False",
        help="""
        Whether to use the relative (non-TransformerXL)-type
        positional embedding in the permutation transformer.
        """)
    parser.add_argument(
        '--decoder_pos_embedding', type=str, default="False",
        help="""
        Whether to use sinusoid position embedding for the
        autoregressive decoder. For Transformer-INDIGO
        this is set to False.
        """)
    parser.add_argument(
        '--embedding_align_coeff', type=float, default=0.0,
        help="""
        The coefficient of embedding alignment loss 
        between the permutation transformer and the decoder.
        This parameter is used for VOI training only.         
        """)    
    parser.add_argument(
        '--kl_coeff', type=float, default=1.0,
        help="""
        Kl divergence coefficient beta
        where the loss is beta * KL((X+eps)/self.temperature || eps/temp_prior),
        where eps is Gumbel noise.
        This parameter is used for VOI training only.
        """)
    parser.add_argument(
        '--kl_log_linear', type=float, default=-1,
        help="""
        If this value > 0, decrease the log coefficient of kl
        linearly as training proceeds until the value given.
        This parameter is used for VOI training only.        .
        """)
    parser.add_argument(
        '--action_refinement', type=int, default=1,
        help="""
        The number of actions (permutations, orderings) to sample
        per training data.
        The actual batch dimension
        has length batch_size * action_refinement.
        This parameter is used for VOI training only.
        """)
    parser.add_argument(
        '--alternate_training', nargs='+', type=int,
        help="""
        If two integers x and y are given, 
        then train the decoder and fix the permutation transformer for x iterations, 
        and then train the permutation transformer and fix the decoder for y iterations.
        Repeat this process until the end of training.
        """)
    parser.add_argument(
        '--use_ppo', action='store_true',
        help="""
        For policy gradient, whether to use PPO.
        This parameter is used for VOI training only.
        """)
    parser.add_argument(
        '--dataset', type=str, default='captioning', 
        choices=['captioning', 'wmt', 'django', 'gigaword'],
        help="""
        Type of dataset.
        'captioning' = coco, flickr;
        'wmt' = machine translation;
        'django' = NL2Code;
        'gigaword' = text summarization
        """)  
    parser.add_argument(
        '--decoder_training_scheme', type=str, default='all', choices=['best', 'all'],
        help="""
        Whether to train decoder with the best permutation ("best")
        or all sampled permutations from the permutation transformer ("all").
        This parameter is used for VOI training only.
        """)    
    parser.add_argument(
        '--parallel_strategy', type=str, default='nccl', choices=['nccl', 'hierarchy'],
        help="""
        tf.distribute.MirroredStrategy options.
        'nccl' = NcclAllReduce;
        'hierarchy' = HierarchicalCopyAllReduce
        """)
    parser.add_argument(
        '--reward_std', type=str, default="False",
        help="""
        For policy gradient, whether to standardize the reward.
        This parameter is used for VOI training only.
        """)
    
    args = parser.parse_args()
    args.share_embedding = (args.share_embedding == "True")
    args.pt_positional_attention = (args.pt_positional_attention == "True")
    args.pt_relative_embedding = (args.pt_relative_embedding == "True")
    args.decoder_pos_embedding = (args.decoder_pos_embedding == "True")
    args.reward_std = (args.reward_std == "True")
    print(args)  
        
    #tf.profiler.experimental.server.start(6009)

    assert not (args.pt_positional_attention and args.pt_relative_embedding)
    if args.dataset == 'captioning':
        assert args.first_layer == 'region'
    elif args.dataset in ['wmt', 'django']:
        assert args.first_layer == 'discrete'
        
    if args.alternate_training is not None:
        assert len(args.alternate_training) == 2
    if args.policy_gradient != 'none':
        assert args.action_refinement >= 1
    if args.policy_gradient == 'with_bvn':
        raise NotImplementedError
    if args.policy_gradient == 'none':
        assert args.action_refinement == 1
        assert not args.use_ppo
    if args.alternate_training:
        assert args.policy_gradient == 'without_bvn'
        
    assert args.warmup >= 0
    assert '.h5' == args.model_ckpt[-3:], "Please save the model in hdf5 format"
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

#     Different tf.distribute.Strategy
#     if args.parallel_strategy == 'nccl':
#         strategy = tf.distribute.MirroredStrategy(
#             cross_device_ops=tf.distribute.NcclAllReduce())
#     elif args.parallel_strategy == 'hierarchy':
#         strategy = tf.distribute.MirroredStrategy(
#             cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())       
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    vocabs = []
    for vfile in args.vocab_file:
        with tf.io.gfile.GFile(vfile, "r") as f:
            vocabs.append(Vocabulary([x.strip() for x in f.readlines()],
                               unknown_word="<unk>",
                               unknown_id=1))

    with strategy.scope():
        
        def get_src_tgt_embedding():
            if args.dataset == 'captioning':
                assert len(vocabs) == 1
                return tf.keras.layers.Embedding(
                            91, args.embedding_size), \
                       tf.keras.layers.Embedding(
                            vocabs[0].size(), args.embedding_size) 
            elif args.dataset == 'wmt':
                assert len(vocabs) == 1
                emb = tf.keras.layers.Embedding(
                            vocabs[0].size(), args.embedding_size)       
                return emb, emb
            elif args.dataset == 'django':
                assert len(vocabs) == 1
                emb = tf.keras.layers.Embedding(
                            vocabs[0].size(), args.embedding_size)
                return emb, emb
            elif args.dataset == 'gigaword':
                assert len(vocabs) == 1
                emb = tf.keras.layers.Embedding(
                            vocabs[0].size(), args.embedding_size)
                return emb, emb            
            
        model_src_embedding, model_tgt_embedding = get_src_tgt_embedding()  
        if args.share_embedding:
            pt_src_embedding = model_src_embedding
            pt_tgt_embedding = model_tgt_embedding
        else:
            pt_src_embedding, pt_tgt_embedding = get_src_tgt_embedding() 
        
        # The decoder autoregressive transformer's "call" function
        # is dummy by design and is only used for initializing the parameters.
        # Do not call model(inputs) to get the results. 
        # Instead, use the model's loss function and the beam/greedy search function
        # to obtain results        
        model = Transformer(vocabs[-1].size(),
                            args.embedding_size,
                            args.heads,
                            args.num_layers,
                            model_src_embedding,
                            model_tgt_embedding,
                            queries_dropout=args.queries_dropout,
                            keys_dropout=args.keys_dropout,
                            values_dropout=args.values_dropout,
                            causal=True,
                            logits_per_slot=2,
                            first_layer=args.first_layer,
                            final_layer=args.final_layer,
                            decoder_pos_emb=args.decoder_pos_embedding,
                            dataset=args.dataset,
                            label_smoothing=args.label_smoothing)

        if args.order == 'soft':
            # The permutation transformer can directly be 
            # called on the inputs i.e. order(inputs)
            order = PermutationTransformer(args.embedding_size,
                                           args.heads,
                                           args.num_layers,
                                           args.policy_gradient,
                                           pt_src_embedding,
                                           pt_tgt_embedding,
                                           queries_dropout=0.1,
                                           keys_dropout=0.1,
                                           values_dropout=0.1,
                                           first_layer=args.first_layer,
                                           pg_final_layer=args.pt_pg_type,
                                           pt_positional_attention=args.pt_positional_attention,
                                           pt_relative_embedding=args.pt_relative_embedding,
                                           dataset=args.dataset,
                                           hungarian_op_path=args.hungarian_op_path)

    train_dataset(args.train_folder,
                  args.batch_size,
                  args.beam_size,
                  args.num_epochs,
                  model,
                  args.model_ckpt,
                  args.save_interval,
                  order if args.order == 'soft' else args.order,
                  vocabs,
                  strategy,
                  args.dataset,
                  args.policy_gradient,
                  args.reward_std,
                  args.pt_pg_type,
                  args.decoder_pretrain,
                  args.decoder_init_lr,
                  args.pt_init_lr,
                  args.lr_schedule,
                  args.warmup,
                  args.kl_coeff,
                  args.kl_log_linear,
                  args.action_refinement,
                  args.alternate_training,
                  args.use_ppo,
                  args.embedding_align_coeff,
                  args.decoder_training_scheme)

from voi.core.sess_util import get_session
from voi.core.train import train_dataset
from voi.nn.transformer import Transformer
from voi.nn.baseline_transformer import BaselineTransformer
from voi.nn.permutation_transformer import PermutationTransformer
from voi.process.captions import Vocabulary
import tensorflow as tf
import argparse

# tf.compat.v1.disable_eager_execution()
# sess = get_session()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_folder', type=str, default='tfrecords')
    parser.add_argument(
        '--batch_size', type=int, default=16)
    parser.add_argument(
        '--beam_size', type=int, default=3)
    parser.add_argument(
        '--vocab_file', type=str, nargs='+')
    parser.add_argument(
        '--num_epochs', type=int, default=10)
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/nsds.h5')
    parser.add_argument(
        '--embedding_size', type=int, default=256)
    parser.add_argument(
        '--share_embedding', type=str, default="True")
    parser.add_argument(
        '--heads', type=int, default=4)
    parser.add_argument(
        '--num_layers', type=int, default=2)
    parser.add_argument(
        '--queries_dropout', type=float, default=0.1)
    parser.add_argument(
        '--keys_dropout', type=float, default=0.1)
    parser.add_argument(
        '--values_dropout', type=float, default=0.1)
    parser.add_argument(
        '--label_smoothing', type=float, default=0.0)    
    parser.add_argument(
        '--first_layer', type=str,
        default='region', choices=['region', 'discrete', 'continuous'])
    parser.add_argument(
        '--final_layer', type=str,
        default='indigo', choices=['indigo', 'logits'])
    parser.add_argument(
        '--order', type=str,
        default='soft', choices=['l2r', 'r2l', 'rare', 'common', 'test', 'soft'])
    parser.add_argument(
        '--policy_gradient', type=str,
        default='none', choices=['with_bvn', 'without_bvn', 'none'])   
    parser.add_argument(
        '--decoder_pretrain', type=int, default=-1)    
    parser.add_argument(
        '--decoder_init_lr', type=float, default=0.0001)      
    parser.add_argument(
        '--pt_init_lr', type=float, default=0.00001) 
    parser.add_argument(
        '--lr_schedule', type=str, 
        default='linear', choices=['linear', 'constant'])
    parser.add_argument(
        '--pt_pg_type', type=str,
        default='sinkhorn', choices=['plackett', 'sinkhorn'])
    parser.add_argument(
        '--pt_positional_attention', type=str, default="False")
    parser.add_argument(
        '--pt_relative_embedding', type=str, default="False")
    parser.add_argument(
        '--decoder_pos_embedding', type=str, default="False")
    parser.add_argument(
        '--kl_coeff', type=float, default=1.0)
    parser.add_argument(
        '--kl_log_linear', type=float, default=-1)
    parser.add_argument(
        '--action_refinement', type=int, default=1)
    parser.add_argument(
        '--alternate_training', nargs='+', type=int)
    parser.add_argument(
        '--use_ppo', action='store_true')
    parser.add_argument(
        '--dataset', type=str, default='captioning', choices=['captioning', 'wmt', 'django', 'gigaword'])  
    parser.add_argument(
        '--decoder_training_scheme', type=str, default='all', choices=['best', 'all'])    
    parser.add_argument(
        '--parallel_strategy', type=str, default='nccl', choices=['nccl', 'hierarchy'])
    parser.add_argument(
        '--reward_std', type=str, default="False")
    args = parser.parse_args()
    args.share_embedding = (args.share_embedding == "True")
    args.pt_positional_attention = (args.pt_positional_attention == "True")
    args.pt_relative_embedding = (args.pt_relative_embedding == "True")
    args.decoder_pos_embedding = (args.decoder_pos_embedding == "True")
    args.reward_std = (args.reward_std == "True")
    print(args)  
        
        
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
        
    assert '.h5' == args.model_ckpt[-3:], "Please save the model in hdf5 format"
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    if args.parallel_strategy == 'nccl':
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce())
    elif args.parallel_strategy == 'hierarchy':
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())       
#     strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    vocabs = []
    for vfile in args.vocab_file:
        with tf.io.gfile.GFile(vfile, "r") as f:
            vocabs.append(Vocabulary([x.strip() for x in f.readlines()],
                               unknown_word="<unk>",
                               unknown_id=1))

    with strategy.scope():
        # Since the transformer contains both the logits layer
        # and the pointer layer, which can both be the final layers,
        # do not call model(inputs) directly. Instead, use the 
        # model's loss function and the beam/greedy search function
        # to obtain results
        
        # in the case of translation, model_src_embedding is unused
        # because the vocabulary is shared between src and tgt
        
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
#                 return tf.keras.layers.Embedding(
#                             vocabs[0].size(), args.embedding_size), \
#                        tf.keras.layers.Embedding(
#                             vocabs[1].size(), args.embedding_size) 
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
                                           dataset=args.dataset)

    train_dataset(args.train_folder,
                  args.batch_size,
                  args.beam_size,
                  args.num_epochs,
                  model,
                  args.model_ckpt,
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
                  args.kl_coeff,
                  args.kl_log_linear,
                  args.action_refinement,
                  args.alternate_training,
                  args.use_ppo,
                  args.decoder_training_scheme)
    
# sess.close()

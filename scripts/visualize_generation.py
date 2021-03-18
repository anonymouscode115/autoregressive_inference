from voi.core.visualize_generation import visualize_generation
from voi.nn.transformer import Transformer
from voi.nn.permutation_transformer import PermutationTransformer
from voi.process.captions import Vocabulary
import tensorflow as tf
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--validate_folder', type=str, default='tfrecords')
    parser.add_argument(
        '--ref_folder', type=str, default='captions')
    parser.add_argument(
        '--batch_size', type=int, default=3)
    parser.add_argument(
        '--beam_size', type=int, default=12)
    parser.add_argument(
        '--vocab_file', type=str, nargs='+')
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/nsds.h5')
    parser.add_argument(
        '--hungarian_op_path', type=str, default='./hungarian.so')
    parser.add_argument(
        '--embedding_size', type=int, default=256)
    parser.add_argument(
        '--heads', type=int, default=4)
    parser.add_argument(
        '--num_layers', type=int, default=2)
    parser.add_argument(
        '--first_layer', type=str,
        default='region', choices=['region', 'discrete', 'continuous'])
    parser.add_argument(
        '--final_layer', type=str,
        default='indigo', choices=['indigo', 'logits'])
    parser.add_argument(
        '--order', type=str,
        default='soft', choices=['l2r', 'r2l', 'rare', 'common', 'soft', 'sao'])
    parser.add_argument(
        '--policy_gradient', type=str,
        default='none', choices=['with_bvn', 'without_bvn', 'none'])
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
        '--dataset', type=str,
        default='captioning', choices=['captioning', 'wmt', 'django', 'gigaword'])
    parser.add_argument(
        '--save_path', type=str, default='inspect_generation_order_stats.txt')
    parser.add_argument(
        '--tagger_file', type=str, default='tagger.pkl')

    args = parser.parse_args()
    args.pt_positional_attention = (args.pt_positional_attention == "True")
    args.pt_relative_embedding = (args.pt_relative_embedding == "True")
    args.decoder_pos_embedding = (args.decoder_pos_embedding == "True")
    print(args)

    assert args.model_ckpt[-3:] == '.h5', "Please use hdf5 saved model format"
    assert tf.io.gfile.exists(args.model_ckpt)

    vocabs = []
    for vfile in args.vocab_file:
        with tf.io.gfile.GFile(vfile, "r") as f:
            vocabs.append(Vocabulary([x.strip() for x in f.readlines()],
                                     unknown_word="<unk>",
                                     unknown_id=1))

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())

    with strategy.scope():
        # Since the transformer contains both the logits layer
        # and the pointer layer, which can both be the final layers,
        # do not call model(inputs) directly. Instead, use the
        # model's loss function and the beam/greedy search function
        # to obtain results
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
        pt_src_embedding, pt_tgt_embedding = get_src_tgt_embedding()

        model = Transformer(vocabs[-1].size(),
                            args.embedding_size,
                            args.heads,
                            args.num_layers,
                            model_src_embedding,
                            model_tgt_embedding,
                            queries_dropout=0.,
                            keys_dropout=0.,
                            values_dropout=0.,
                            causal=True,
                            logits_per_slot=2,
                            first_layer=args.first_layer,
                            final_layer=args.final_layer,
                            decoder_pos_emb=args.decoder_pos_embedding,
                            dataset=args.dataset)

        if args.order == 'soft':
            # The permutation transformer can directly be
            # called on the inputs i.e. order(inputs)
            order = PermutationTransformer(args.embedding_size,
                                           args.heads,
                                           args.num_layers,
                                           args.policy_gradient,
                                           pt_src_embedding,
                                           pt_tgt_embedding,
                                           queries_dropout=0.,
                                           keys_dropout=0.,
                                           values_dropout=0.,
                                           first_layer=args.first_layer,
                                           pg_final_layer=args.pt_pg_type,
                                           pt_positional_attention=args.pt_positional_attention,
                                           pt_relative_embedding=args.pt_relative_embedding,
                                           dataset=args.dataset,
                                           hungarian_op_path=args.hungarian_op_path)

    visualize_generation(
        args.validate_folder,
        args.ref_folder,
        args.batch_size,
        args.beam_size,
        model,
        args.model_ckpt,
        order if args.order == 'soft' else args.order,
        vocabs,
        strategy,
        args.policy_gradient,
        args.save_path,
        args.dataset,
        args.tagger_file)

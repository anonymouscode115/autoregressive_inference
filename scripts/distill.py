from voi.core.distill import distill_dataset
from voi.nn.transformer import Transformer
from voi.process.captions import Vocabulary
import tensorflow as tf
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_folder', type=str, default='tfrecords')
    parser.add_argument(
        '--batch_size', type=int, default=3)
    parser.add_argument(
        '--beam_size', type=int, default=12)
    parser.add_argument(
        '--vocab_file', type=str, nargs='+')
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/nsds.h5')
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
        '--dataset', type=str,
        default='captioning', choices=['captioning', 'wmt', 'django', 'gigaword'])
    parser.add_argument(
        '--decoder_pos_embedding', type=bool, default="False")
    parser.add_argument(
        '--save_path', type=str, default='', help="save path for distillation output")    
    args = parser.parse_args()
    args.decoder_pos_embedding = (args.decoder_pos_embedding == "True")
    
    # distillation unavailable for captioning tasks since there are
    # already multiple references
    assert args.dataset != 'captioning'
    
    if args.dataset in ['wmt', 'django', 'gigaword']:
        assert args.first_layer == 'discrete'
        
    assert args.model_ckpt[-3:] == '.h5', "Please use hdf5 saved model format"
    assert tf.io.gfile.exists(args.model_ckpt)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    
    vocabs = []
    for vfile in args.vocab_file:
        with tf.io.gfile.GFile(vfile, "r") as f:
            vocabs.append(Vocabulary([x.strip() for x in f.readlines()],
                               unknown_word="<unk>",
                               unknown_id=1))    

    with strategy.scope():
        def get_src_tgt_embedding():
            if args.dataset == 'wmt':
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

    distill_dataset(args.train_folder,
                    args.batch_size,
                    args.beam_size,
                    model,
                    args.model_ckpt,
                    vocabs,
                    args.dataset,
                    strategy,
                    args.save_path)

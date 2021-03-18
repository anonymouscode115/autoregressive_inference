from voi.data.tfrecords_wmt import create_tfrecord_wmt
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_tfrecord_folder', type=str, default='/home/wmt/train_40/')
    parser.add_argument(
        '--feature_folder', type=str, default='/home/wmt/train_en_de/')
    parser.add_argument(
        '--dataset_type', type=str, default='validation', choices=['train', 'validation', 'test', 'distillation'])    
    parser.add_argument(
        '--samples_per_shard', type=int, default=4096)
    args = parser.parse_args()

    create_tfrecord_wmt(args.out_tfrecord_folder,
                    args.feature_folder,
                    args.dataset_type,
                    args.samples_per_shard)
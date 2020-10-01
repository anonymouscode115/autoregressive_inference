from voi.process.captions_wmt import process_wmt
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_feature_folder', type=str, default='/home/wmt/validation_en_de/')
    parser.add_argument(
        "--data_folder", type=str, default=None)
    parser.add_argument(
        '--vocab_file', type=str, default='/home/wmt/vocab_40.txt')
    parser.add_argument(
        '--max_length', type=int, default=70)
    parser.add_argument(
        '--min_word_frequency', type=int, default=1)
    parser.add_argument(
        '--dataset_type', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument(
        '--one_vocab', action='store_true')    
    args = parser.parse_args()

    process_wmt(args.out_feature_folder,
                     args.data_folder,
                     args.vocab_file,
                     args.max_length,
                     args.min_word_frequency,
                     args.dataset_type,
                     args.one_vocab)

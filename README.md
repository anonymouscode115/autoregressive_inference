# Learning Autoregressive Orderings with Variational Inference

## Installation

To install this package, first download the package from github, then install it using pip.

```bash
git clone git@github.com:{name/voi}
pip install -e voi
```

Install helper packages for word tokenization and part of speech tagging. Enter the following statements into the python interpreter where you have installed our package.

```python
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('universal_tagset')
```

Finally, install the natural language evaluation package that contains several helpful metrics.

```bash
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```

You can now start training a non-sequential model!

## Setup - Captioning

In this section, we will walk you through how to create a training dataset, using COCO 2017 as an example. In the first step, download COCO 2017. Place the annotations at `~/annotations` and the images at `~/train2017` and `~/val2017` for the training and validation set respectively.

Create a part of speech tagger first.

```bash
python scripts/data/create_tagger.py --out_tagger_file tagger.pkl
```

Extract COCO 2017 into a format compatible with our package. There are several arguments that you can specify to control how the dataset is processed. You may leave all arguments as default except `out_caption_folder` and `annotations_file`.

```bash
python scripts/data/extract_coco.py --out_caption_folder ~/captions_train2017 --annotations_file ~/annotations/captions_train2017.json
python scripts/data/extract_coco.py --out_caption_folder ~/captions_val2017 --annotations_file ~/annotations/captions_val2017.json
```

Process the COCO 2017 captions and extract integer features on which to train a non sequential model. There are again several arguments that you can specify to control how the captions are processed. You may leave all arguments as default except `out_feature_folder` and `in_folder`, which depend on where you extracted the COCO dataset in the previous step.

```bash
python scripts/data/process_captions.py --out_feature_folder ~/captions_train2017_features --in_folder ~/captions_train2017 --tagger_file tagger.pkl --vocab_file train2017_vocab.txt --min_word_frequency 5 --max_length 100
python scripts/data/process_captions.py --out_feature_folder ~/captions_val2017_features --in_folder ~/captions_val2017 --tagger_file tagger.pkl --vocab_file train2017_vocab.txt --max_length 100
```

Process images from the COCO 2017 dataset and extract features using a pretrained Faster RCNN FPN backbone from pytorch checkpoint. Note this script will distribute inference across all visible GPUs on your system. There are several arguments you can specify, which you may leave as default except `out_feature_folder` and `in_folder`, which depend on where you extracted the COCO dataset.

```bash
python scripts/data/process_images.py --out_feature_folder ~/train2017_features --in_folder ~/train2017 --batch_size 4
python scripts/data/process_images.py --out_feature_folder ~/val2017_features --in_folder ~/val2017 --batch_size 4
```

Finally, convert the processed features into a TFRecord format for efficient training. Record where you have extracted the COCO dataset in the previous steps and specify `out_tfrecord_folder`, `caption_folder` and `image_folder` at the minimum.

```bash
python scripts/data/create_tfrecords.py --out_tfrecord_folder ~/train2017_tfrecords --caption_folder ~/captions_train2017_features --image_folder ~/train2017_features --samples_per_shard 4096
python scripts/data/create_tfrecords.py --out_tfrecord_folder ~/val2017_tfrecords --caption_folder ~/captions_val2017_features --image_folder ~/val2017_features --samples_per_shard 4096
```

## Setup - Django

For convenience, we ran the script from [NL2code](https://github.com/pcyin/NL2code) to extract the cleaned dataset from [drive](https://drive.google.com/drive/folders/0B14lJ2VVvtmJWEQ5RlFjQUY2Vzg) and place them in `django_data`. Alternatively, you may download raw data from [ase15-django](https://github.com/odashi/ase15-django-dataset) and run `python scripts/data/extract_django.py --data_dir {path to all.anno and all.code)`

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/process_django.py --data_folder ./django_data --vocab_file ./django_data/djangovocab.txt --one_vocab --dataset_type train/dev/test --out_feature_folder ./django_data
CUDA_VISIBLE_DEVICES=0 python scripts/data/create_tfrecords_django.py --out_tfrecord_folder ./django_data --dataset_type train/dev/test --feature_folder ./django_data
```

## Training

You may train several kinds of models using our framework. For example, you can replicate our results and train a non-sequential soft-autoregressive Transformer-InDIGO model using the following command in the terminal.

#### COCO
```bash
# Train with embedding shared between encoder and decoder first
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py --train_folder ~/train2017_tfrecords --batch_size 36 --beam_size 1 --vocab_file train2017_vocab.txt --num_epochs 4 --model_ckpt ckpt_refinement/nsds_coco_voi.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer region --final_layer indigo --order soft --policy_gradient without_bvn --decoder_pretrain -1 --decoder_init_lr 0.0001 --pt_init_lr 0.00001 --kl_coeff 0.3 --action_refinement 4 --share_embedding True --lr_schedule constant --pt_pg_type sinkhorn --pt_relative_embedding False --reward_std False --pt_positional_attention True --label_smoothing 0.1" > nohup_coco_voi.txt

# Then train with embedding separated
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py --train_folder ~/train2017_tfrecords --batch_size 36 --beam_size 1 --vocab_file train2017_vocab.txt --num_epochs 15 --model_ckpt ckpt_refinement/nsds_coco_voi.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer region --final_layer indigo --order soft --policy_gradient without_bvn --decoder_pretrain -1 --decoder_init_lr 0.00005 --pt_init_lr 0.000005 --kl_coeff 0.3  --kl_log_linear 0.03 --action_refinement 4 --share_embedding False --lr_schedule constant --pt_pg_type sinkhorn --pt_relative_embedding False --reward_std False --pt_positional_attention True --label_smoothing 0.1" > nohup_coco_voi.txt
```

#### Django
```bash
# Train with embedding shared between encoder and decoder first
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py --train_folder django_data/train --batch_size 36 --beam_size 1 --vocab_file django_data/djangovocab.txt --num_epochs 50 --model_ckpt ckpt_refinement/nsds_django_voi.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer discrete --final_layer indigo --order soft --policy_gradient without_bvn --action_refinement 4 --use_ppo --kl_coeff 0.3 --decoder_init_lr 0.0001 --pt_init_lr 0.00001 --lr_schedule constant --label_smoothing 0.1 --dataset django --pt_positional_attention True --share_embedding True --reward_std False" > nohup_nsds_django_voi.txt

# Then train with embedding separated
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py --train_folder django_data/train --batch_size 36 --beam_size 1 --vocab_file django_data/djangovocab.txt --num_epochs 200 --model_ckpt ckpt_refinement/nsds_django_voi.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer discrete --final_layer indigo --order soft --policy_gradient without_bvn --action_refinement 4 --use_ppo --kl_coeff 0.3 --kl_log_linear 0.03 --decoder_init_lr 0.00003 --pt_init_lr 0.000003 --lr_schedule constant --label_smoothing 0.1 --dataset django --pt_positional_attention True --share_embedding False --reward_std False" > nohup_nsds_django_voi.txt

# It can be helpful to anneal entropy to 0 to make encoder converge to single ordering for each data
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py --train_folder django_data/train --batch_size 36 --beam_size 1 --vocab_file django_data/djangovocab.txt --num_epochs 50 --model_ckpt ckpt_refinement/nsds_django_voi.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer discrete --final_layer indigo --order soft --policy_gradient without_bvn --action_refinement 4 --use_ppo --kl_coeff 0.03 --kl_log_linear 0.003 --decoder_init_lr 0.00003 --pt_init_lr 0.000003 --lr_schedule constant --label_smoothing 0.1 --dataset django --pt_positional_attention True --share_embedding False --reward_std False" > nohup_nsds_django_voi.txt
```

## Validation / Test

You may evaluate a trained model with the following command. If you are not able to install the `nlg-eval` package, the following command will still run and print captions for the validation set, but it will not calculate evaluation metrics.

#### COCO

```bash
python scripts/validate.py --validate_folder ~/val2017_tfrecords --ref_folder ~/captions_val2017 --batch_size 32 --beam_size 2 --vocab_file train2017_vocab.txt --model_ckpt ckpt/nsds_coco_voi_ckpt.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer region --final_layer indigo
```


#### Django

```bash
python scripts/validate.py --validate_folder django_data/(dev/test) --ref_folder "" --batch_size 4 --beam_size 10 --vocab_file django_data/djangovocab.txt --model_ckpt ckpt/nsds_django_voi_ckpt.h5 --embedding_size 512 --heads 8 --num_layers 6 --first_layer discrete --final_layer indigo
```

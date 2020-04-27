# transformers-keras

Transformer-based models implemented in tensorflow 2.x(Keras).

## Models

- [x] Transformer
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
  * Here is a tutorial from tensorflow:[Transformer model for language understanding](https://www.tensorflow.org/beta/tutorials/text/transformer)
- [x] BERT
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [x] ALBERT
  * [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)


## Transformer

You should process your data in tfrecord format. Modify this script `transformers_keras/utils/transformer_tfrecord_generator.py` as you need.

```bash
python -m transformers_keras.run_transformer \
    --mode train \
    --model_config config/transformer_model_config.json \
    --data_config config/transformer_data_config.json
```


## BERT

Use your own data to pretrain a BERT model.

```bash
python -m transformers_keras.run_bert \
    --mode train \
    --model_config config/bert_model_config.json \
    --data_config config/bert_data_config.json
```

> Tips:
> You need prepare your data to tfrecord format. You can use this script: [create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)


## ALBERT

You should process your data in tfrecord format. Modify this script `transformers_keras/utils/bert_tfrecord_custom_generator.py` as you need.

```bash
python -m transformers_keras.run_bert \
    --mode train \
    --model_config config/albert_model_config.json \
    --data_config config/albert_data_config.json
```
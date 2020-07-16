# transformers-keras

![Python package](https://github.com/luozhouyang/transformers-keras/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/transformers-keras.svg)](https://badge.fury.io/py/transformers-keras)
[![Python](https://img.shields.io/pypi/pyversions/transformers-keras.svg?style=plastic)](https://badge.fury.io/py/transformers-keras)

Transformer-based models implemented in tensorflow 2.x(Keras).

[中文文档](README_ZH.md) | [English]

## Contents

- [transformers-keras](#transformers-keras)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Models](#models)
  - [Transformer](#transformer)
  - [BERT](#bert)
    - [Train a new BERT model](#train-a-new-bert-model)
    - [Load a pretrained BERT model](#load-a-pretrained-bert-model)
  - [ALBERT](#albert)
    - [Train a new ALBERT model](#train-a-new-albert-model)
    - [Load a pretrained ALBERT model](#load-a-pretrained-albert-model)


## Installation

```bash
pip install -U transformers-keras
```

## Models

- [x] Transformer
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
  * Here is a tutorial from tensorflow:[Transformer model for language understanding](https://www.tensorflow.org/beta/tutorials/text/transformer)
- [x] BERT
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [x] ALBERT
  * [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)


## Transformer

Train a new transformer:

```python
from transformers_keras import TransformerTextFileDatasetBuilder
from transformers_keras import TransformerDefaultTokenizer
from transformers_keras import TransformerRunner


src_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_src.txt')
tgt_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_tgt.txt')
dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer)

model_config = {
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'src_vocab_size': src_tokenizer.vocab_size,
    'tgt_vocab_size': tgt_tokenizer.vocab_size,
}

runner = TransformerRunner(model_config, dataset_builder, model_dir='/tmp/transformer')

train_files = [('testdata/train.src.txt','testdata/train.tgt.txt')]
runner.train(train_files, epochs=10, callbacks=None)

```

## BERT

You can use `BERT` models in two ways:

* [Train a new BERT model](#train-a-new-bert-model)
* [Load a pretrained BERT model](#load-a-pretrained-bert-model)


### Train a new BERT model

Use your own data to pretrain a BERT model.

```python
from transformers_keras import BertForPretrainingModel

model_config = {
    'max_positions': 128,
    'num_layers': 6,
    'vocab_size': 21128,
}

model = BertForPretrainingModel(**model_config)
```

### Load a pretrained BERT model


```python
from transformers_keras import BertForPretrainingModel

# download the pretrained model and extract it to some path
PRETRAINED_BERT_MODEL = '/path/to/chinese_L-12_H-768_A-12'

model = BertForPretrainingModel.from_pretrained(PRETRAINED_BERT_MODEL)
```

After building the model, you can train the model with your own data.

Here is an example:

```python
from transformers_keras import BertTFRecordDatasetBuilder

builder = BertTFRecordDatasetBuilder(max_sequence_length=128, record_option='GZIP')

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
model.compile(optimizer='adam', loss=loss, metrics=[metric])
model(model.dummy_inputs())
model.summary()

train_files = ['testdata/bert_custom_pretrain.tfrecord']
train_dataset = builder.build_train_dataset(train_files, batch_size=32)
model.fit(train_dataset, epochs=2)
```

## ALBERT

You can use `ALBERT` model in two ways:
* [Train a new ALBERT model](#train-a-new-albert-model)
* [Load a pretrained ALBERT model](#load-a-pretrained-albert-model)


### Train a new ALBERT model
You should process your data to tfrecord format. Modify this script `transformers_keras/utils/bert_tfrecord_custom_generator.py` as you need.


```python
from transformers_keras import AlbertForPretrainingModel

# ALBERT has the same data format with BERT
dataset_builder = BertTFRecordDatasetBuilder(
    max_sequence_length=128, record_option='GZIP', train_repeat_count=100, eos_token='T')

model_config = {
    'max_positions': 128,
    'num_layers': 6,
    'num_groups': 1,
    'num_layers_each_group': 1,
    'vocab_size': 21128,
}

model = AlbertForPretrainingModel(**model_config)
```

### Load a pretrained ALBERT model


```python
from transformers_keras import AlbertForPretrainingModel

# download the pretrained model and extract it to some path
PRETRAINED_BERT_MODEL = '/path/to/zh_albert_large'

model = AlbertForPretrainingModel.from_pretrained(PRETRAINED_BERT_MODEL)
```

After building the model, you can train this model with your own data.

Here is an example:

```python
from transformers_keras import BertTFRecordDatasetBuilder

builder = BertTFRecordDatasetBuilder(max_sequence_length=128, record_option='GZIP')

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
model.compile(optimizer='adam', loss=loss, metrics=[metric])
model(model.dummy_inputs())
model.summary()

train_files = ['testdata/bert_custom_pretrain.tfrecord']
train_dataset = builder.build_train_dataset(train_files, batch_size=32)
model.fit(train_dataset, epochs=2)
```


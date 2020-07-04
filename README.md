# transformers-keras

![Python package](https://github.com/luozhouyang/transformers-keras/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/transformers-keras.svg)](https://badge.fury.io/py/transformers-keras)
[![Python](https://img.shields.io/pypi/pyversions/transformers-keras.svg?style=plastic)](https://badge.fury.io/py/transformers-keras)

Transformer-based models implemented in tensorflow 2.x(Keras).

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

Use your own data to pretrain a BERT model.

```python
from transformers_keras import BertTFRecordDatasetBuilder
from transformers_keras import BertRunner


dataset_builder = BertTFRecordDatasetBuilder(
    max_sequence_length=128, record_option='GZIP', train_repeat_count=100, eos_token='T')

model_config = {
    'max_positions': 128,
    'num_layers': 6,
    'vocab_size': 21128,
}

runner = BertRunner(model_config, dataset_builder, model_dir='models/bert')

train_files = ['testdata/bert_custom_pretrain.tfrecord']
runner.train(train_files, epochs=10, callbacks=None)

```
Tips:
>
> You need prepare your data to tfrecord format. You can use this script: [create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)
>
> You can subclass `transformers_keras.tokenizers.BertTFRecordDatasetBuilder` to parse custom tfrecord examples as you need.


### Load the pretrained model

You can use an `Adapter` to load pretrained models.

Here is an example.

```python
from transformers_keras.adapters import BertAdapter

# download the pretrained model and extract it to some path
PRETRAINED_BERT_MODEL = '/path/to/chinese_L-12_H-768_A-12'

adapter = BertAdapter()
model = adapter.adapte(PRETRAINED_BERT_MODEL)

print('model inputs: {}'.format(model.inputs))
print('model outputs: {}'.format(model.outputs))

```

will print:

```bash
model inputs: [<tf.Tensor 'input_ids:0' shape=(None, 512) dtype=int32>, <tf.Tensor 'segment_ids:0' shape=(None, 512) dtype=int32>, <tf.Tensor 'input_mask:0' shape=(None, 512) dtype=int32>]
model outputs: [<tf.Tensor 'predictions/Identity:0' shape=(512, 21128) dtype=float32>, <tf.Tensor 'relations/Identity:0' shape=(2,) dtype=float32>]
```

Then, you can use this model to do anything you want!


## ALBERT

You should process your data to tfrecord format. Modify this script `transformers_keras/utils/bert_tfrecord_custom_generator.py` as you need.


```python
from transformers_keras import BertTFRecordDatasetBuilder
from transformers_keras import AlbertRunner

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

runner = AlbertRunner(model_config, dataset_builder, model_dir='models/albert')

train_files = ['testdata/bert_custom_pretrain.tfrecord']
runner.train(train_files, epochs=10, callbacks=None)

```
